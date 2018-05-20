// copyright Luca Istrate, Andrei Medar

#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // TODO(you)
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
    unordered_map<int, int> hashTable;

    is_leaf = true;

    if (is_single_class) {
        result = samples[0][0];
        return;
    }

    for (int i = 0; i < samples.size(); i++) {
        hashTable[samples[i][0]]++;
    }
    int digit, maxApparitionCount = 0;
    for (auto it = hashTable.begin(); it != hashTable.end(); it++) {
        if (it->second > maxApparitionCount) {
            maxApparitionCount = it->second;
            digit = it->first;
        }
    }
    result = digit;
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensionNum) {
    // TODO(you)
    // Intoarce cea mai buna dimensiune si valoare de split dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)

    int splitIndex = -1, splitValue = -1;
    int crtValue;
    float IG, IGmax = 0.0f, aux;
    vector<int> dimensionValues, leftSplit, rightSplit;
    pair<vector<int>, vector<int>> splits;
    for (int i = 0; i < dimensionNum.size(); i++) {
        crtValue = 0;
        dimensionValues = compute_unique(samples, dimensionNum[i]);
        for (int j = 0; j < dimensionValues.size(); j++)
            crtValue += dimensionValues[j];
        crtValue /= dimensionValues.size();
        splits = get_split_as_indexes(samples, dimensionNum[i], crtValue);
        leftSplit = splits.first;
        rightSplit = splits.second;
        if (leftSplit.size() && rightSplit.size()) {
            aux = (float)leftSplit.size();
            IG = get_entropy_by_indexes(samples, leftSplit) * aux;
            aux = (float)rightSplit.size();
            IG += get_entropy_by_indexes(samples, rightSplit) * aux;
            IG /= samples.size();
            IG = get_entropy(samples) - IG;
            if (IG > IGmax) {
                IGmax = IG;
                splitValue = crtValue;
                splitIndex = dimensionNum[i];
            }
        }
    }
    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // TODO(you)
    // Antreneaza nodul curent si copiii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv
    if (same_class(samples)) {
        this->make_leaf(samples, true);
        return;
    }
    vector<int> dimensions = random_dimensions(samples[0].size());
    pair<int, int> data = find_best_split(samples, dimensions);
    split_index = data.first;
    split_value = data.second;
    if (split_index == -1 && split_value == -1) {
        this->make_leaf(samples, false);
        return;
    }
    pair<vector<vector<int>>, vector<vector<int>>> splits;
    splits = split(samples, split_index, split_value);
    left->train(splits.first);
    right->train(splits.second);
}

int Node::predict(const vector<int> &image) const {
    // TODO(you)
    // Intoarce rezultatul prezis de catre decision tree
    int prediction = result;
    if (!is_leaf) {
        if (image[split_index] > split_value)
            prediction = right->predict(image);
        else
            prediction = left->predict(image);
    }
    return prediction;
}

bool same_class(const vector<vector<int>> &samples) {
    // TODO(you)
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
    for (int i = 0; i < samples.size() - 1; i++) {
        if (samples[i][0] != samples[i + 1][0]) {
            return false;
        }
    }
    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // Intoarce entropia testelor primite
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // TODO(you)
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)
    float entropy = 0.0f;
    int n = index.size();
    unordered_map<int, int> apparitions;
    for (int i = 0; i < n; i++)
        apparitions[samples[index[i]][0]]++;
    for (auto it = apparitions.begin(); it != apparitions.end(); it++) {
        float p = (float)(it->second / n);
        if (it->second)
            entropy -= p * (std::log(p) / std::log(2));
    }
    return entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // TODO(you)
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
    vector<int> uniqueValues;
    unordered_map<int, int> hashTable;

    for (int i = 0; i < samples.size(); i++) {
        if (!hashTable[samples[i][col]]){
            uniqueValues.push_back(samples[i][col]);
            hashTable[samples[i][col]] = 1;
        }
    }
    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // TODO(you)
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
    vector<int> left, right;

    for (int i = 0; i < samples.size(); i++) {
        if (samples[i][split_index] <= split_value) {
            left.push_back(i);
        } else {
            right.push_back(i);
        }
    }
    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // TODO(you)
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
    vector<int> rez;
    unordered_map<int, int> hashTable;

    std::uniform_int_distribution<int> distribution(0, size);
    std::random_device randomDevice;

    int dimensionNum = (int)sqrt(size), i = 0;
    while (i < dimensionNum) {
        int randomValue = distribution(randomDevice);
        if (randomValue == 0 || randomValue == size) {
            continue;
        } else {
            if (hashTable[randomValue]) {
                continue;
            } else {
                rez.push_back(randomValue);
                hashTable[randomValue] = 1;
            }
        }
        i++;
    }
    return rez;
}
