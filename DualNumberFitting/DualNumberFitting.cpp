#include <iostream>
#include <vector>
#include "NN.h"
/*

This project trains a neural network using autodifferentiation with dual numbers
it's based on an idea presented in the following link:
https://blog.demofox.org/2017/03/09/how-to-train-neural-networks-with-backpropagation/
Where the author mentions an alternative to traditional back propagation auto differentiation with dual numbers

The network is trained by subtracting a gradient vector from the weights and biases of the network
The data it is trained on produces an 'XOR' operation on it's input data after around 1000 iteration

*/
void XOR() {
    std::vector<int> desc{ 2, 2, 1 }; //describes the network architecture
    NN a = NN(desc); //initializes the network
    std::vector<std::vector<float>> vc //input data
    {
        {0, 0},
        {0, 0.75},
        {0.75, 0},
        {0.75, 0.75}
    };
    std::vector<std::vector<float>> vo //desired output data
    {
        {0},
        {0.75},
        {0.75},
        {0}
    };
    //training loop
    //calculates the average gradient vector and subtracts it from the weights and biases of the network
    a.train(1000, 0.4, vc, vo);
    std::cout << " \nNetwork desc: 2 input, 2 hidden, 1 output\n\n";
    std::cout << "\t::XOR::\n";
    for (int y = 0; y < vc.size(); y++)
    {
        std::cout << vc[y] << ":\t" << a.forwardF(vc[y]) << "\n"; //print results 
    }
}
std::vector<float> f(std::vector<float> k)
{
    return k;
}

std::vector<float> square(std::vector<float> k)
{
    std::vector<float> o(k.size());
    for (int x = 0; x < k.size(); x++)
    {
        o[x] = k[x] * k[x];
    }
    return o;
}
std::vector<float> mul(std::vector<float> k)
{
    std::vector<float> o(1);
    o[0] = k[0] * k[1];
    return o;
}
//This function is trained so that it's output is the same as it's input
//the hidden layer is smaller than the input/output layers, so the data is being compressed
//into a single neuron
void Compress()
{
   
    int numIterations = 750;
    float learningRate = 0.4;
    int numSamples = 250;
    std::vector<int> desc{ 3, 2, 3 }; //describes the network architecture
    int numOutOfSample = 10;

    int inpLength = desc[0];
    NN a = NN(desc); //initializes the network
    std::vector<std::vector<float>> inp(numSamples);
    std::vector<std::vector<float>> outp(numSamples);
    fill2DVectorWithFunc(&inp, &outp, f, inpLength, 0, 1);
   
    float C = a.train(numIterations, learningRate, inp, outp);

    std::vector<std::vector<float>> inpTest(numOutOfSample);
    fill2DVector(&inpTest, inpLength, 0, 1);
    for (int y = 0; y < inpTest.size(); y++)
    {
        std::cout << inpTest[y] << ":\t" << a.forwardF(inpTest[y]) << "\n"; //print results 
    }
    std::cout << C;
}

void Continuous()
{
    int numIterations = 750;
    float learningRate = 0.05;
    int numSamples = 450;
    std::vector<int> desc{ 2, 2, 1 }; //describes the network architecture
    int numOutOfSample = 10;

    int inpLength = desc[0];
    NN a = NN(desc); //initializes the network
    std::vector<std::vector<float>> inp(numSamples); //input training data
    std::vector<std::vector<float>> outp(numSamples); //output training data
    fill2DVectorWithFunc(&inp, &outp, mul, inpLength, 0, 1); //fill output data with samples of a continuous function

    float C = a.train(numIterations, learningRate, inp, outp); //train

    std::vector<std::vector<float>> inpTest(numOutOfSample);
    fill2DVector(&inpTest, inpLength, 0, 1);
    for (int y = 0; y < inpTest.size(); y++)
    {
        std::cout << inpTest[y] << ":\t (x1*x2) approx: " << a.forwardF(inpTest[y]) << "\treal: " << inpTest[y][0]*inpTest[y][1] << "\n"; //print results 
    }
    std::cout << C;
}

int main()
{  
    XOR();
    //Compress();
    //Continuous();
}

