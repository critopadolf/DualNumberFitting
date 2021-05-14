#include "NN.h"
//constructor for the network, takes in an integer vector whose length is the number of layers, and each integer is the number of neurons in that layer
NN::NN(std::vector<int> desc) {
    int px = 0;
    numWeights = 0;
    numBiases = 0;
    for (int& lc : desc) {
        vals.push_back(std::vector<CDualNumber>(lc)); //vals stores the current value of each neuron, initialize to 0
        if (px != 0)
        {
            std::vector<float> k = std::vector<float>(lc); //create a float vector for each neuron layer of specified length
            fillVector(&k, -1.0, 1.0); //fill biases with rand vals from -1 to 1
            biases.push_back(k); //push to biases vector

            std::vector< std::vector<float> > w = std::vector< std::vector<float> >(lc); //create 2D weight vector

            fill2DVector(&w, px, -1.0, 1.0); //fill weight vector with random numbers from -1 to 1
            weights.push_back(w); //tack on this neurons weights to the main weight vector
            numWeights += px * lc; //count the total number of weights
            numBiases += lc; //total number of biases (# of neurons that are not in the input layer)
        }
        px = lc; //the size of the previous neuron layer, used to determine the number of weights for each neuron in the current layer
    }
    numVars = numWeights + numBiases; //total number of variables that describe the network, equals the length of the gradient vector
    for (int k = 0; k < vals.size(); k++)
    {
        for (int u = 0; u < vals[k].size(); u++)
        {
            vals[k][u] = CDualNumber(numVars);
        }
    }
}

//feed forward the network with a given input and a desired output for that network, converting
//the input and network variables to dual numbers to determine the gradient of the cost function for the given input/output combination

CDualNumber NN::forwardDual(std::vector<float> inp, std::vector<float> outp)
{
    int weightIndex = 0; //determines which epsilon value to set to 1 for a given weight
    int biasIndex = numWeights; //same as the above, but offset to set different epsilons to 1 for biases based on the number of weights
    CDualNumber C(numVars); //the Cost dual number (gradient) to be returned from this function

    for (int j = 0; j < inp.size(); j++)
    {
        vals[0][j] = CDualNumber(numVars, inp[j]); //load input into first neuron layer
    }
    for (int x = 0; x < weights.size(); x++) {
        //x == neuron layer (skipping 1st layer)
        for (int y = 0; y < weights[x].size(); y++)
        {
            //y == current neuron in the layer
            vals[x + 1][y] = CDualNumber(numVars, biases[x][y], biasIndex); //the current neurons' value starts at the bias of that neuron, reinstantiated as a dual number
            for (int z = 0; z < weights[x][y].size(); z++)
            {
                //z == weight for each connection to current neuron
                vals[x + 1][y] = vals[x + 1][y] + CDualNumber(numVars, weights[x][y][z], weightIndex) * vals[x][z]; //each weight is multiplied by the value of each of the previous neurons and added the the total value
                weightIndex++; //increment epsilon index for the weights
            }
            vals[x + 1][y] = activate(vals[x + 1][y]); //the sinusoidal activation function is used on the value of the current neuron
            biasIndex++; //increment epsilon index for the biases
        }
    }
    int lne = vals.size() - 1; //index for the last layer of the network
    //this loop calculates the average cost of the network by dividing the dual number output of each output neuron by the total number of output neurons and summing them
    for (int k = 0; k < outp.size(); k++)
    {
        C = C + pow(CDualNumber(numVars, outp[k]) - vals[lne][k], 2) / CDualNumber(numVars, vals[lne].size()); //cost function (y - v)^2 divided by number of output neurons
    }
    //std::cout << C.m_real << "\n"; //the real component of the dual number C is the current average cost for this given input
    return C; //return the cost to be averaged with the cost of the other inputs and passed to the subGrad function
}


//subtract a gradient vector (g) scaled by a learning rate (lr) from the weights and biases of the network
void NN::subGrad(CDualNumber g, float lr)
{

    int weightIndex = 0;
    int biasIndex = numWeights;
    //for each weight and bias, subtract their assigned epsilon value from the dual number g (after scaling the dual number by the constan, lr)
    g = g * CDualNumber(numVars, lr); //scale the gradient vector by the learning rate
    for (int x = 0; x < weights.size(); x++)
    {
        //x == neuron layer (skipping 1st layer)
        for (int y = 0; y < weights[x].size(); y++)
        {
            //y == current neuron in the layer
            biases[x][y] -= g.m_dual[biasIndex]; //subtract gradient from biases
            for (int z = 0; z < weights[x][y].size(); z++)
            {
                //z == weight for each connection to current neuron
                weights[x][y][z] -= g.m_dual[weightIndex]; //subtract gradient from weights
                weightIndex++;
            }
            biasIndex++;
        }
    }

}

//this function is the same as the other forward function but doesn't use dual numbers or a cost function
//it just gets the real output of the network, so it is faster and used after training is done
std::vector<float> NN::forwardF(std::vector<float> inp)
{
    int lne = vals.size() - 1;
    int ol = vals[lne].size();
    std::vector<float> outp(ol);
    for (int j = 0; j < inp.size(); j++)
    {
        vals[0][j].m_real = inp[j]; //load input into first neuron layer
    }
    for (int x = 0; x < weights.size(); x++) {
        //x == neuron layer (skipping 1st layer)
        for (int y = 0; y < weights[x].size(); y++)
        {
            //y == current neuron in the layer
            vals[x + 1][y].m_real = biases[x][y]; //CDualNumber(float f, size_t variableIndex)
            for (int z = 0; z < weights[x][y].size(); z++)
            {
                //z == weight for each connection to current neuron
                vals[x + 1][y].m_real = vals[x + 1][y].m_real + weights[x][y][z] * vals[x][z].m_real;
            }
            vals[x + 1][y].m_real = activate(vals[x + 1][y]).m_real;
        }
    }
    for (int x = 0; x < ol; x++)
    {
        outp[x] = vals[lne][x].m_real; //load real values of output neurons into the output vector
    }
    return outp;
}
CDualNumber NN::activate(CDualNumber x)
{
    /*
    //leaky relu
    if (x.m_real >= 0)
    {
        return x;
    }
    else
    {
        return CDualNumber(numVars,0.001) * x;
    }
    */
    //sinusoidal
    return sin(x);
}

float NN::train(int n, float learnRate, std::vector<std::vector<float>> inp, std::vector<std::vector<float>> outp)
{
    if (inp[0].size() != weights[0][0].size())
    {
        std::cout << "input size mismatch\n";
        std::cout << "input given: " << inp[0].size() << " input neurons: " << weights[0].size();
        exit(EXIT_FAILURE);
    }
    if (outp[0].size() != weights[weights.size()-1].size())
    {
        std::cout << "output size mismatch\n";
        std::cout << "output given: " << outp[0].size() << " output neurons: " << weights[weights.size() - 1].size();
        exit(EXIT_FAILURE);
    }
    float m;
    for (int x = 0; x < n; x++)
    {
        CDualNumber C(numVars, 0); //instantiate the gradient of the Cost function
        for (int y = 0; y < inp.size(); y++)
        {
            C = C + forwardDual(inp[y], outp[y]); //add up every gradient for the given input
        }
        C = C / CDualNumber(numVars, inp.size()); //take average of gradient vector
        subGrad(C, learnRate); //move in direction of average gradient vector
        m = C.m_real;
        std::cout << x << "/" << n << "\t" << m << "\n";
    }
    return m;
}


