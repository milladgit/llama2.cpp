#pragma once

#include <vector>

struct ProbIndex {
    float prob;
    int index;
}; // struct used when sorting probabilities during top-p sampling

struct Sampler {
    int vocab_size;
//    ProbIndex* probindex; // buffer used in top-p sampling
    std::vector<ProbIndex> probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;

    void build(int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
        vocab_size = vocab_size;
        temperature = temperature;
        topp = topp;
        rng_state = rng_seed;
        // buffer only used with nucleus sampling; may not need but it's ~small
//        probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
        probindex = std::vector<ProbIndex>(vocab_size);
    }

};

//void free_sampler(Sampler* sampler) {
//    free(sampler->probindex);
//}
//
