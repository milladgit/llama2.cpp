#pragma once

#include "tokenizer.h"
#include "transformer.h"

#include <sstream>

int str_lookup(char *str, std::vector<TokenIndex>& sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab.data(), vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax(Tensor<float>& x, int size) {
    // only 1D support for now!
    assert(x.getShape().size() == 1);
    softmax(x.getData(), size);
}

void rmsnorm(Tensor<float>& o, const Tensor<float>& x, const Tensor<float>& weight, int size) {

    auto print = [](const Tensor<float>& t) -> std::string {
        if (t.getShape().size() == 0) {
            return "";
        }
        std::stringstream ss;
        ss << "[";
        for(auto x : t.getShape()) {
            ss << x << ", ";
        }
        ss << "]";
        return ss.str();
    };

    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x(j) * x(j);
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o(j) = weight(0, j) * (ss * x(j));
    }
}

void matmul(Tensor<float>& xout, const Tensor<float>& x, const Tensor<float>& w, int n, int d) {

    auto print = [](const Tensor<float>& t) -> std::string {
        if (t.getShape().size() == 0) {
            return "";
        }
        std::stringstream ss;
        ss << "[";
        for(auto x : t.getShape()) {
            ss << x << ", ";
        }
        ss << "]";
        return ss.str();
    };

    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
//            val += w(i * n + j) * x(j);
            val += w(0, i, j) * x(j);
        }
        xout(i) = val;
    }
}

template <typename DataType>
Tensor<float> forward(Transformer<DataType>* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    auto& x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

    // copy the token embedding into x
//    float* content_row = w->token_embedding_table + token * dim;
    auto* content_row = w->token_embedding_table + token * dim;
    memcpy(x.getData(), content_row.getData(), dim * sizeof(float));

    std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // attention rmsnorm
//        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        rmsnorm(s->xb, x, w->rms_att_weight.cropWithOffset({l, 0ULL}), dim);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
//        s->k = s->key_cache + loff + pos * kv_dim;
//        s->v = s->value_cache + loff + pos * kv_dim;
        s->k = s->key_cache.cropWithOffset({l, (unsigned long long)pos, 0ULL});
        s->v = s->value_cache.cropWithOffset({l, (unsigned long long)pos, 0ULL});

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // qkv matmuls for this position
//        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
//        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
//        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        matmul(s->q, s->xb, w->wq.cropWithOffset({l, 0ULL, 0ULL}), dim, dim);
        matmul(s->k, s->xb, w->wk.cropWithOffset({l, 0ULL, 0ULL}), dim, kv_dim);
        matmul(s->v, s->xb, w->wv.cropWithOffset({l, 0ULL, 0ULL}), dim, kv_dim);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                auto& vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec(i);
                float v1 = vec(i+1);
                vec(i)   = v0 * fcr - v1 * fci;
                vec(i+1) = v0 * fci + v1 * fcr;
            }
        }

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
//            float* q = s->q + h * head_size;
            auto q = s->q.cropWithOffset({h});
            // attention scores for this head
//            float* att = s->att + h * p->seq_len;
            auto att = s->att.template cropWithOffset({h, 0});
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache.getData() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q(i) * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att(t) = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

            // weighted sum of the values, store back into xb
//            float* xb = s->xb + h * head_size;
            auto xb = s->xb.template cropWithOffset({h});
            memset(xb.getData(), 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache.getData() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att(t);
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb(i) += a * v[i];
                }
            }
        }

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // final matmul to get the output of the attention
//        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        matmul(s->xb2, s->xb, w->wo.cropWithOffset({l, 0ULL, 0ULL}), dim, dim);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x(i) += s->xb2(i);
        }

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // ffn rmsnorm
//        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        rmsnorm(s->xb, x, w->rms_ffn_weight.cropWithOffset({l, 0ULL}), dim);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;


        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
//        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
//        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb, s->xb, w->w1.cropWithOffset({l, 0ULL, 0ULL}), dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3.cropWithOffset({l, 0ULL, 0ULL}), dim, hidden_dim);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
//            float val = s->hb[i];
            float val = s->hb(i);
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2(i);
//            s->hb[i] = val;
            s->hb(i) = val;
        }

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

        // final matmul to get the output of the ffn
//        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        matmul(s->xb, s->hb, w->w2.cropWithOffset({l, 0ULL, 0ULL}), hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x(i) += s->xb(i);
        }
    }

    std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << std::endl;

    // classifier into logits
//    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    matmul(s->logits, x, w->wcls.cropWithOffset({p->dim}), p->dim, p->vocab_size);
    return s->logits;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, std::vector<int>& tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab.size()) {
        // lazily malloc and sort the vocabulary
//        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        t->sorted_vocab = std::vector<TokenIndex>(t->vocab_size);
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i].data();
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab.data(), t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
//    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
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
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
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
    for (char *c = text; *c != '\0'; c++) {

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
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
//            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            std::stringstream ss;
            ss << t->vocab[tokens[i]].data() << t->vocab[tokens[i+1]].data();
//            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            int id = str_lookup((char*)ss.str().c_str(), t->sorted_vocab, t->vocab_size);
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

//    free(str_buffer);
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

template<typename DataType>
void generate(Transformer<DataType> *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
//    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    std::vector<int> prompt_tokens(strlen(prompt)+3); // +3 for '\0', ?BOS, ?EOS
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
//        float* logits = forward(transformer, token, pos);
        auto logits = forward(transformer, token, pos);

//        // advance the state machine
//        if (pos < num_prompt_tokens - 1) {
//            // if we are still processing the input prompt, force the next prompt token
//            next = prompt_tokens[pos + 1];
//        } else {
//            // otherwise sample the next token from the logits
//            next = sample(sampler, logits);
//        }
//        pos++;
//
//        // data-dependent terminating condition: the BOS (=1) token delimits sequences
//        if (next == 1) { break; }
//
//        // print the token as string, decode it with the Tokenizer object
//        char* piece = decode(tokenizer, token, next);
//        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
//        fflush(stdout);
//        token = next;

//        // init the timer here because the first iteration can be slower
//        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
//
//    // report achieved tok/s (pos-1 because the timer starts after first iteration)
//    if (pos > 1) {
//        long end = time_in_ms();
//        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
//    }

//    free(prompt_tokens);
}

