// copyright Luca Istrate, Andrei Medar
#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <unordered_map>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // TODO(you)
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
    vector<vector<int>> ret;
    std::unordered_map<int, int> hashTable;

    std::uniform_int_distribution<int> distribution(0, samples.size());
    std::random_device randomDevice;

    int randomValue, i = 0;
    while (i < num_to_return) {
        randomValue = distribution(randomDevice);
        if (randomValue == samples.size()) {
            continue;
        } else {
            if (hashTable[randomValue]) {
                continue;
            } else {
                ret.push_back(samples[randomValue]);
                hashTable[randomValue] = 1;
            }
        }
        i++;
    }
    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // TODO(you)
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
    std::unordered_map<int, int> apparitions;
    for (int i = 0; i < num_trees; i++)
        apparitions[trees[i].predict(image)]++;
    int prediction, maxApparitionCount = 0;
    for (auto it = apparitions.begin(); it != apparitions.end(); it++)
        if (it->second > maxApparitionCount) {
            maxApparitionCount = it->second;
            prediction = it->first;
        }
    return prediction;
}
