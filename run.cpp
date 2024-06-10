/* Inference for Llama-2 Transformer model in C++ */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// C++ headers
#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>


// ----------------------------------------------------------------------------
// Transformer model

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

/// This struct is loaded from the file!
/// As a result, we keep the pointers and we will not promote them to use STD types.
/// The data type used in this structure is mandated by the file as they are loaded from the file.
template <typename T = float>
struct TransformerWeights {
    // token embedding table
    T* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    T* rms_att_weight; // (layer, dim) rmsnorm weights
    T* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    T* wq; // (layer, dim, n_heads * head_size)
    T* wk; // (layer, dim, n_kv_heads * head_size)
    T* wv; // (layer, dim, n_kv_heads * head_size)
    T* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    T* w1; // (layer, hidden_dim, dim)
    T* w2; // (layer, dim, hidden_dim)
    T* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    T* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    T* wcls;
};

/// This struct is allocated!
template <typename T>
struct RunState {
    // current wave of activations
    std::vector<T> x; // activation at current time stamp (dim,)
    std::vector<T> xb; // same, but inside a residual branch (dim,)
    std::vector<T> xb2; // an additional buffer just for convenience (dim,)
    std::vector<T> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<T> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<T> q; // query (dim,)
    // The followings (k, v) are only used for pointer arithmetic operations.
    // No real data are allocated for them.
    T* k; // key   ---> upper bound for shape: (dim,)
    T* v; // value ---> upper bound for shape: (dim,)
    std::vector<T> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<T> logits; // output logits
    // kv cache
    std::vector<T> key_cache;   // (layer, seq_len, dim)
    std::vector<T> value_cache; // (layer, seq_len, dim)

    RunState() = default;

    void allocate(int dim, int hidden_dim, int cache_size, int att_size, int logits_size) {
        x.resize(dim, 0);
        xb.resize(dim, 0);
        xb2.resize(dim, 0);
        hb.resize(hidden_dim, 0);
        hb2.resize(hidden_dim, 0);
        q.resize(dim, 0);
        key_cache.resize(cache_size, 0);
        value_cache.resize(cache_size, 0);
        att.resize(att_size, 0);
        logits.resize(logits_size, 0);
    }

};

template <typename T>
struct Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights<T> weights; // the weights of the model
    RunState<T> state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
};

template<typename T>
void malloc_run_state(RunState<T>* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->allocate(p->dim,
                p->hidden_dim,
                p->n_layers * p->seq_len * kv_dim,
                p->n_heads * p->seq_len,
                p->vocab_size);
}

template<typename T>
void memory_map_weights(TransformerWeights<T> *w, Config* p, float* ptr, int shared_weights) {
    static_assert(std::is_same_v<T, float>);

    // at the moment, the file is loaded with `float` values in them!
    static_assert(std::is_same_v<std::remove_pointer_t<decltype(ptr)>, float>);

    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

template<typename T>
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights<T>* weights,
                     int* fd, T** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (T*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    T* weights_ptr = *data + sizeof(Config)/sizeof(T);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

template <typename T>
void build_transformer(Transformer<T> *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path,
                    &t->config,
                    &t->weights,
                    &t->fd,
                    &t->data,
                    &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

template <typename T>
void free_transformer(Transformer<T>* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
}



// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer
template <typename T>
void rmsnorm(T* o, T* x, T* weight, int size) {
    // Please add implementation of other types!
    static_assert(std::is_same_v<T, float>);

    // calculate sum of squares
    T ss = T(0);
    #pragma omp parallel for private(j) reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += T(1e-5f);
    ss = T(1) / std::sqrt(ss);
    // normalize and scale
    #pragma omp parallel for private(j)
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

template <typename T>
void softmax(T* x, int size) {
    // Please add implementation of other types!
    static_assert(std::is_same_v<T, float>);

    // find max value (for numerical stability)
    T max_val = x[0];
    #pragma omp parallel for private(i) reduction(max:max_val)
    for (int i = 1; i < size; i++) {
        max_val = std::max(max_val, x[i]);
    }
    // exp and sum
    T sum = T(0);
    #pragma omp parallel for private(i) reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    T sum_inv = T(1) / sum;
    // normalize
    #pragma omp parallel for private(i)
    for (int i = 0; i < size; i++) {
        x[i] *= sum_inv;
    }
}

template <typename T>
void matmul(T* xout, T* x, T* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        T val = T(0);
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// INTERMEDIATE_TYPE refers to the type that we usually use to store intermediate values in it.
// It is useful in cases where we are dealing with fp16 precision values.
template <typename T, typename INTERMEDIATE_TYPE = float>
T* forward(Transformer<T>* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights<T>* w = &transformer->weights;
    RunState<T>* s = &transformer->state;
    T *x = s->x.data();
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    T* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(T));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb.data(), x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        const int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache.data() + loff + pos * kv_dim;
        s->v = s->value_cache.data() + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q.data(), s->xb.data(), w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb.data(), w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb.data(), w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            const int head_dim = i % head_size;
            const INTERMEDIATE_TYPE freq = INTERMEDIATE_TYPE(1) / std::pow(INTERMEDIATE_TYPE(10000), head_dim / (INTERMEDIATE_TYPE)head_size);
            const INTERMEDIATE_TYPE val = pos * freq;
            const INTERMEDIATE_TYPE fcr = cosf(val);
            const INTERMEDIATE_TYPE fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                T* vec = v == 0 ? s->q.data() : s->k; // the vector to rotate (query or key)
                const T v0 = vec[i];
                const T v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            T* q = s->q.data() + h * head_size;
            // attention scores for this head
            T* att = s->att.data() + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                T* k = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                T score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= std::sqrt(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            T* xb = s->xb.data() + h * head_size;
            memset(xb, 0, head_size * sizeof(T));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                const T* v =  s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                const T a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2.data(), s->xb.data(), w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb.data(), x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb.data(), s->xb.data(), w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2.data(), s->xb.data(), w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            T val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (T) (1.0f / (1.0f + std::exp(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb.data(), s->hb.data(), w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits.data(), x, w->wcls, p->dim, p->vocab_size);

    return s->logits.data();
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct TokenIndex {
    // Kept as pointer because it is used to pass strings!
    // More efficient to use it as it is!
    char *str;
    int id;
};

template <typename T>
struct Tokenizer {
    std::vector<std::vector<char>> vocab;
    std::vector<T> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

    Tokenizer() = default;
};

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

template <typename T>
void build_tokenizer(Tokenizer<T>* t, const char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab.resize(vocab_size);
    t->vocab_scores.resize(vocab_size);

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i].resize(len + 1);
        if (fread(t->vocab[i].data(), len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

template <typename T>
char* decode(Tokenizer<T>* t, int prev_token, int token) {
    char* piece = t->vocab[token].data();
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, const std::vector<TokenIndex>& sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    auto res = (TokenIndex *) bsearch(&tok,
                                      (void*)sorted_vocab.data(),
                                      vocab_size,
                                      sizeof(TokenIndex),
                                      compare_tokens);
    return res != nullptr ? res->id : -1;
}

template <typename T>
void encode(Tokenizer<T>* t, const char *text, int8_t bos, int8_t eos, std::vector<int>& tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == nullptr) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab.empty()) {
        t->sorted_vocab.resize(t->vocab_size);
        for (int i = 0; i < t->vocab_size; i++) {
            assert(t->vocab[i].size() > 0);
            t->sorted_vocab[i].str = t->vocab[i].data();
            t->sorted_vocab[i].id = i;

            assert(t->sorted_vocab[i].str != nullptr);
        }

        qsort(t->sorted_vocab.data(), t->sorted_vocab.size(), sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    std::vector<char> str_buffer(t->max_token_length*2 +1 +2);
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer.data(), t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            // TODO: Use string_stream;
            // BUT, I think this might even be more efficient as we do no need to reallocated multiple times!
            sprintf(str_buffer.data(), "%s%s", t->vocab[tokens[i]].data(), t->vocab[tokens[i+1]].data());
            int id = str_lookup(str_buffer.data(), t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}


// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// TODO: Making float a template param
//template<typename T>
struct ProbIndex {
    float prob;
    int index;
}; // struct used when sorting probabilities during top-p sampling

template<typename T>
struct Sampler {
    static_assert(std::is_same_v<T, float>);

    int vocab_size;
    std::vector<ProbIndex> probindex; // buffer used in top-p sampling
    T temperature;
    T topp;
    unsigned long long rng_state;

    Sampler() = default;

    void allocate(int vocab_size, T temperature, T topp, unsigned long long rng_seed) {
        this->vocab_size = vocab_size;
        this->temperature = temperature;
        this->topp = topp;
        this->rng_state = rng_seed;
        // buffer only used with nucleus sampling; may not need but it's ~small
        probindex.resize(vocab_size);
    }
};

template <typename T>
int sample_argmax(T* probabilities, int n) {
    // return the index that has the highest probability
    return std::max_element(probabilities, probabilities + n) - probabilities;
}

template <typename T>
int sample_mult(T* probabilities, int n, T coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    T cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare_prob_index(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

template <typename T>
int sample_topp(T* probabilities, int n, T topp, ProbIndex* probindex, T coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const T cutoff = (T(1) - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob_index);

    // truncate the list where cumulative probability exceeds topp
    T cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    T r = coin * cumulative_prob;
    T cdf = T(0);
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

template<typename T>
void build_sampler(Sampler<T>* sampler, int vocab_size, T temperature, T topp, unsigned long long rng_seed) {
    sampler->allocate(vocab_size, temperature, topp, rng_seed);
}


unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

template <typename T>
int sample(Sampler<T>* sampler, T* logits) {
    // for new dtypes, implement random_f32 and random_u32
    static_assert(std::is_same_v<T, float>);

    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == T(0)) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        T coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits,
                               sampler->vocab_size,
                               sampler->topp,
                               sampler->probindex.data(),
                               coin);
        }
    }
    return next;
}


// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

template <typename T>
void generate(Transformer<T> *transformer, Tokenizer<T> *tokenizer, Sampler<T> *sampler, const char *prompt, int steps) {

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    std::vector<int> prompt_tokens((strlen(prompt)+3) ); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}


// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.
template<typename T>
void chat(Transformer<T> *transformer, Tokenizer<T> *tokenizer, Sampler<T> *sampler,
          const char *cli_user_prompt, const char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    std::array<char, 512> system_prompt;
    std::array<char, 512> user_prompt;
    std::stringstream rendered_prompt;
    int num_prompt_tokens = 0;
    // Seems like `1152` is arbitrary! So, we can change it if we want!
    std::vector<int> prompt_tokens(1152);
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == nullptr) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ",
                               system_prompt.data(),
                               system_prompt.size());
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt.data(), cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != nullptr) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt.data(), cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt.data(), user_prompt.size());
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                // TODO: We can use https://github.com/fmtlib/fmt
                rendered_prompt << "[INST] <<SYS>>\n" << system_prompt.data() << "\n<</SYS>>\n\n" << user_prompt.data() << " [/INST]";
            } else {
                // TODO: We can use https://github.com/fmtlib/fmt
                rendered_prompt << "[INST] " << user_prompt.data() << " [/INST]";
            }

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt.str().c_str(), 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        T* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = nullptr;  // e.g. out/model.bin
    const char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    const char *prompt = nullptr;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    const char *mode = "generate";    // generate|chat
    char *system_prompt = nullptr; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }
    if(prompt == nullptr) { prompt = ""; }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    using DataType = float;

    // build the Transformer via the model .bin file
    Transformer<DataType> transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer<DataType> tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler<DataType> sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer,
                 &tokenizer,
                 &sampler,
                 prompt,
                 steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer,
             &tokenizer,
             &sampler,
             prompt,
             system_prompt,
             steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // file handles cleanup
    free_transformer(&transformer);
    return 0;
}
#endif
