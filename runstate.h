#pragma once

#include "tensor.h"
#include "config.h"

struct RunState {
    // current wave of activations
    Tensor<float> x; // activation at current time stamp (dim,)
    Tensor<float> xb; // same, but inside a residual branch (dim,)
    Tensor<float> xb2; // an additional buffer just for convenience (dim,)
    Tensor<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor<float> q; // query (dim,)
    Tensor<float> k; // key (dim,)
    Tensor<float> v; // value (dim,)
    Tensor<float> att; // buffer for scores/attention values (n_heads, seq_len)
    Tensor<float> logits; // output logits
    // kv cache
    Tensor<float> key_cache;   // (layer, seq_len, dim)
    Tensor<float> value_cache; // (layer, seq_len, dim)

    RunState() {}

    RunState(Config* p) {
        using ULL = Tensor<int>::ShapeT;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        x = Tensor<float>({(ULL)p->dim});
        xb = Tensor<float>({(ULL)p->dim});
        xb2 = Tensor<float>({(ULL)p->dim});
        hb = Tensor<float>({(ULL)p->hidden_dim});
        hb2 = Tensor<float>({(ULL)p->hidden_dim});
        q = Tensor<float>({(ULL)p->dim});
        key_cache = Tensor<float>({(ULL)p->n_layers, (ULL)p->seq_len, (ULL)kv_dim});
        value_cache = Tensor<float>({(ULL)p->n_layers, (ULL)p->seq_len, (ULL)kv_dim});
        att = Tensor<float>({(ULL)p->n_heads, (ULL)p->seq_len});
        logits = Tensor<float>({(ULL)p->vocab_size});
    }
};


