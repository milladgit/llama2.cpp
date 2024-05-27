#pragma once

#include "tensor.h"

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    // token embedding table
    Tensor<float> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    Tensor<float> rms_att_weight; // (layer, dim) rmsnorm weights
    Tensor<float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Tensor<float> wq; // (layer, dim, n_heads * head_size)
    Tensor<float> wk; // (layer, dim, n_kv_heads * head_size)
    Tensor<float> wv; // (layer, dim, n_kv_heads * head_size)
    Tensor<float> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    Tensor<float> w1; // (layer, hidden_dim, dim)
    Tensor<float> w2; // (layer, dim, hidden_dim)
    Tensor<float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    Tensor<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    Tensor<float> wcls;
};


