#pragma once

#include <string>
#include <vector>
#include <iostream>

struct TokenIndex {
    char *str;
//    std::vector<char> str;
    int id;
};

int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


struct Tokenizer {
//    char** vocab;
    std::vector<std::vector<char>> vocab;
//    float* vocab_scores;
    std::vector<float> vocab_scores;
//    TokenIndex *sorted_vocab;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

    void build(char* tokenizer_path, int vocab_size) {
        // i should have written the vocab_size into the tokenizer file... sigh
        vocab_size = vocab_size;
        // malloc space to hold the scores and the strings
//    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
        vocab = std::vector<std::vector<char>>(vocab_size);
//    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
        vocab_scores = std::vector<float>(vocab_size);
//    t->sorted_vocab = NULL; // initialized lazily
        for (int i = 0; i < 256; i++) {
            byte_pieces[i * 2] = (unsigned char)i;
            byte_pieces[i * 2 + 1] = '\0';
        }
        // read in the file
        FILE *file = fopen(tokenizer_path, "rb");
        if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        int len;
        for (int i = 0; i < vocab_size; i++) {
//        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
            if (fread(&vocab_scores[i], sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
            if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
//        t->vocab[i] = (char *)malloc(len + 1);
            vocab[i] = std::vector<char>(len + 1);
//        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
            if (fread(vocab[i].data(), len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

};


//void free_tokenizer(Tokenizer* t) {
//    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
//    free(t->vocab);
//    free(t->vocab_scores);
//    free(t->sorted_vocab);
//}


