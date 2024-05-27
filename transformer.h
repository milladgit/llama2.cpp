#pragma once

#include "runstate.h"

template <typename DataType>
struct Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    DataType* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes

    void build(char* checkpoint_path) {
        // read in the Config and the Weights from the checkpoint
        read_checkpoint(checkpoint_path, &config, &weights, &fd, &data, &file_size);
        // allocate the RunState buffers
//        malloc_run_state(&t->state, &t->config);
        state = RunState(&config);
    }

private:
    void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                         int* fd, DataType** data, ssize_t* file_size) {
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
        *data = (DataType*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
        float* weights_ptr = *data + sizeof(Config)/sizeof(float);
        memory_map_weights(weights, config, weights_ptr, shared_weights);
    }

    void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
        using ULL = Tensor<int>::ShapeT;

        int head_size = p->dim / p->n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        ULL n_layers = p->n_layers;

        w->token_embedding_table = Tensor<float>(ptr, {(ULL)p->vocab_size, (ULL)p->dim});
        ptr += p->vocab_size * p->dim;

        w->rms_att_weight = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim});
        ptr += n_layers * p->dim;

        w->wq = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim, (ULL)(p->n_heads * head_size)});
        ptr += n_layers * p->dim * (p->n_heads * head_size);

        w->wk = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim, (ULL)(p->n_kv_heads * head_size)});
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

        w->wv = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim, (ULL)(p->n_kv_heads * head_size)});
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

        w->wo = Tensor<float>(ptr, {(ULL)n_layers, (ULL)(p->n_heads * head_size), (ULL)p->dim});
        ptr += n_layers * (p->n_heads * head_size) * p->dim;

        w->rms_ffn_weight = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim});
        ptr += n_layers * p->dim;

        w->w1 = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim, (ULL)p->hidden_dim});
        ptr += n_layers * p->dim * p->hidden_dim;

        w->w2 = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->hidden_dim, (ULL)p->dim});
        ptr += n_layers * p->hidden_dim * p->dim;

        w->w3 = Tensor<float>(ptr, {(ULL)n_layers, (ULL)p->dim, (ULL)p->hidden_dim});
        ptr += n_layers * p->dim * p->hidden_dim;

        w->rms_final_weight = Tensor<float>(ptr, {(ULL)p->dim});
        ptr += p->dim;

        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)

        w->wcls = Tensor<float>(shared_weights ? w->token_embedding_table.getData() : ptr, {}); // Problematic as we do not know its shape!
    }

};

