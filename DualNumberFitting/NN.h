#pragma once
#include <stdio.h>
#include <cmath>
#include <array>
#include <algorithm>
#include <vector>
#include "CDualNumber.h"
#include "vectorRandom.cpp"


class NN
{
public:
    std::vector<std::vector<std::vector<float>>> weights; //3D weights vector (layer, neuron, weights)
    std::vector<std::vector<float>> biases; //2D bias vector (layer, neuron), no biases for first layer
    std::vector<std::vector<CDualNumber>> vals; //neuron values (layer, neuron value)
    int numWeights = 0; //number of weights in the network
    int numBiases = 0; //number of biases in the network
    size_t numVars = 0; //number of weights+biases in the network

    
    //constructor for the network, takes in an integer vector whose length is the number of layers, and each integer is the number of neurons in that layer
    NN(std::vector<int> desc);

    //feed forward the network with a given input and a desired output for that network, converting
    //the input and network variables to dual numbers to determine the gradient of the cost function for the given input/output combination
    CDualNumber forwardDual(std::vector<float> inp, std::vector<float> outp);

    //subtract a gradient vector (g) scaled by a learning rate (lr) from the weights and biases of the network
    void subGrad(CDualNumber g, float lr);

    //this function is the same as the other forward function but doesn't use dual numbers or a cost function
    //it just gets the real output of the network, so it is faster and used after training is done
    std::vector<float> forwardF(std::vector<float> inp);

    float train(int n, float learnRate, std::vector<std::vector<float>> inp, std::vector<std::vector<float>> outp);

    CDualNumber activate(CDualNumber x);
};

