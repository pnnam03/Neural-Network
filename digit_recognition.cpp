#include <bits/stdc++.h>
using namespace std;

/*************** struct Connection ***************/
struct Connection
{
    double weight;
    double deltaWeight;
};

int rep = 0;
/***************** class Neuron *****************/
class Neuron
{
public:
    double weightedInput;
    double outputVal;
    double gradient;
    vector<Connection> outputWeight;
};

/****************** typedef Layer ***************/
typedef vector<Neuron> Layer;

/******************** class Net	*****************/
class Net
{
public:
    Net(vector<int> &topology);
    void feedForward(vector<double> &inputVals);
    void backPropagation(vector<double> &targetVals);
    void updateWeight();
    void getResult(vector<double> &resultVals);
    void saveNet(vector<int> &topology);

    void calOutputGradient(vector<double> &targetVals);
    void calHiddenGradient(Layer &curLayer, Layer &nextLayer);
    void calDeltaWeight(Layer &curLayer, Layer &nextLayer);
    double randomWeight()
    {
        return rand() / double(RAND_MAX);
    }
    double activationFunction(double x);
    double activationDerivative(double x);

    vector<Layer> layerList;
    static double eta;
    static double alpha;
    static double BIAS_VALUE;
};

double Net::eta = 0.15;  //
double Net::alpha = 0.5; // momentum
double Net::BIAS_VALUE = 1;

Net::Net(vector<int> &topology)
{
    for (int i : topology)
    {
        layerList.push_back(Layer());
        /*
            each Layer includes i Neuron and 1 extra Bias Neuron
            push a (i+1)-size Layer to Net
        */
        for (int j = 0; j < i + 1; ++j)
            layerList.back().push_back(Neuron());

        // outputValue of Bias neuron is always a constraint
        layerList.back().back().outputVal = BIAS_VALUE;
    }

    for (int i = 0; i < topology.size() - 1; ++i)
    {
        for (int j = 0; j < layerList[i].size(); ++j)
        {
            Neuron &curNeuron = layerList[i][j];
            for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
            {
                curNeuron.outputWeight.push_back(Connection());
                curNeuron.outputWeight.back().weight = randomWeight();
            }
        }
    }
}

void Net::feedForward(vector<double> &inputVals)
{
    for (int i = 0; i < inputVals.size(); ++i)
        layerList[0][i].outputVal = inputVals[i];

    for (int i = 1; i < layerList.size(); ++i)
    {
        // calculate the outputVal of this Layer
        // using the outputVal of the prev
        Layer &curLayer = layerList[i];
        Layer &preLayer = layerList[i - 1];
        // outputVals of bias neurons have to remain the same => curlayer.size() - 1
        for (int j = 0; j < curLayer.size() - 1; ++j)
        {
            double sumWeightedInput = 0;

            for (int k = 0; k < preLayer.size(); ++k)
            {
                double weight = preLayer[k].outputWeight[j].weight;
                sumWeightedInput += preLayer[k].outputVal * weight;
            }

            curLayer[j].weightedInput = sumWeightedInput;
            curLayer[j].outputVal = activationFunction(sumWeightedInput);
            //if (rep % 100 == 0)
            //    printf("%lf\n",curLayer[j].outputVal);
        }
    }

    // vector<double> newOutputVal = softmaxFunction(layerList.back());
    // Layer &lastLayer = layerList.back();
    // for (int i = 0; i < newOutputVal.size(); ++i)
    //     lastLayer[i].outputVal = newOutputVal[i];
}

void Net::backPropagation(vector<double> &targetVals)
{
    calOutputGradient(targetVals);
    for (int i = layerList.size() - 2; i >= 0; --i)
        calHiddenGradient(layerList[i], layerList[i + 1]);
    updateWeight();
}

void Net::updateWeight()
{
    // calculate deltaWeight
    // gradient descent
    for (int i = 0; i < layerList.size() - 1; ++i)
        calDeltaWeight(layerList[i], layerList[i + 1]);
}

void Net::getResult(vector<double> &resultVals)
{
    resultVals.clear();
    Layer &lastLayer = layerList.back();
    for (int i = 0; i < lastLayer.size() - 1; ++i)
        resultVals.push_back(lastLayer[i].outputVal);
}

void Net::saveNet(vector<int> &topology)
{
    ofstream outfile;
    outfile.open("net.dat");

    for (int i = 0; i < topology.size(); ++i)
        outfile << topology[i] << " ";
    outfile << "\n";

    for (int i = 0; i < layerList.size() - 1; ++i)
    {
        for (int j = 0; j < layerList[i].size(); ++j)
        {
            for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
                outfile << layerList[i][j].outputWeight[k].weight << " ";
            outfile << "\n";
        }
    }

    outfile.close();
}

void Net::calOutputGradient(vector<double> &targetVals)
{
    Layer &outputLayer = layerList.back();
    // calculate the d(cost) / d(outputVal)
    double dif = 0;
    int numNeurons = outputLayer.size();
    for (int i = 0; i < numNeurons - 1; ++i)
    {
        // cost = (f(weightedInput) - targetVal) ^ 2
        // => d(cost) / d(weightedInput) = 2 * (outputVal - targetVal) * f'(weightedInput)
        // f(weightedInput) = outputVal
        outputLayer[i].gradient = (outputLayer[i].outputVal - targetVals[i]) *
                                  activationDerivative(outputLayer[i].weightedInput);

        dif += (outputLayer[i].outputVal - targetVals[i]) *
               (outputLayer[i].outputVal - targetVals[i]);
    }
    rep++;
    if (rep > 59000 || rep % 1000 == 0)
        cerr << dif << "\n";
}

void Net::calHiddenGradient(Layer &curLayer, Layer &nextLayer)
{
    for (int i = 0; i < curLayer.size(); ++i)
    {
        double sumGradient = 0.0;
        for (int j = 0; j < nextLayer.size() - 1; ++j)
        {
            double weight = curLayer[i].outputWeight[j].weight;
            sumGradient += nextLayer[j].gradient * weight;
        }
        curLayer[i].gradient = sumGradient * activationDerivative(curLayer[i].weightedInput);
    }
}

void Net::calDeltaWeight(Layer &curLayer, Layer &nextLayer)
{
    for (int i = 0; i < curLayer.size(); ++i)
        for (int j = 0; j < nextLayer.size() - 1; ++j)
        {
            Connection &con = curLayer[i].outputWeight[j];
            double oldDeltaWeight = con.deltaWeight;
            double newDeltaWeight = nextLayer[j].gradient * curLayer[i].outputVal;

            con.deltaWeight = eta * newDeltaWeight + alpha * oldDeltaWeight;

            con.weight -= con.deltaWeight;
        }
}

double Net::activationFunction(double x)
{
    // sigmoid
    // return 1.0 / (1 + exp(-x));

    // tanh
//    return tanh(x/15);

    // ReLU
     return max(x, 0.0)/10;
}

double Net::activationDerivative(double x)
{
    // derivative of sigmoid
    // double sigmoid = activationFunction(x);
    // return sigmoid * (1 - sigmoid);

    // derivative of tanh
//    double val = tanh(x/15);
//    return 1.0 - val * val;

    // derivative of ReLU
     return 1.0*(x > 0)/10;
}


/***************** Main program here ********************/

const int NUM_TRAINING_TEST = 60000;

int label[NUM_TRAINING_TEST + 1];
int image[NUM_TRAINING_TEST + 1][30][30];

void readTrainingData();
int n, rowCount, colCount;

void softmaxFunction(vector<double> &val){
    double maxVal = -1e9;
    double sum = 0;

    for (int i = 0; i < val.size(); ++i)
        sum = sum + exp(val[i]);

    for (int i = 0; i < val.size(); ++i)
        val[i] = exp(val[i]) / sum;
}
int main()
{
    srand(time(NULL));
    //freopen("bin","w",stdout);
    // get the training data
    readTrainingData();

    // initialize the net
    vector<int> topology;
    topology.push_back(rowCount * colCount);
    topology.push_back(10);
    topology.push_back(10);
    topology.push_back(10);

    Net myNet = Net(topology);

    // training
    vector<double> inputVals;
    vector<double> targetVals;
    vector<double> outputVals;

    for (int x = 0; x < rowCount; ++x)
        for (int y = 0; y < colCount; ++y)
            inputVals.push_back(0);
    for (int j = 0; j < 10; ++j)
        targetVals.push_back(0);

    for (int i = 0; i < n; ++i)
    {
        // cerr << "i = " << i << "\n";
        for (int x = 0; x < rowCount; ++x)
            for (int y = 0; y < colCount; ++y)
            {
                inputVals[x * colCount + y] = 1.0 * image[i][x][y] / 255;
                // cout << inputVals[x*colCount + y] << "\n";
            }
        for (int j = 0; j < targetVals.size(); ++j)
            targetVals[j] = 0;
        targetVals[label[i]] = 1;

        myNet.feedForward(inputVals);
        myNet.backPropagation(targetVals);
    }

    myNet.saveNet(topology);

    // testing
    int cnt = 0;
    ofstream outfile;
    outfile.open("out.txt");
    for (int i = n; i < n + 100; ++i)
    {
        for (int x = 0; x < rowCount; ++x)
            for (int y = 0; y < colCount; ++y)
                inputVals[x * colCount + y] = 1.0 * image[i][x][y] / 255;

        for (int j = 0; j < targetVals.size(); ++j)
            targetVals[j] = 0;
        targetVals[label[i]] = 1;

        myNet.feedForward(inputVals);
        myNet.getResult(outputVals);

        //softmaxFunction(outputVals);

        outfile << "Test #" << i - n << "\n";
        outfile << "Target: ";
        for (int j = 0; j < targetVals.size(); ++j)
            outfile
                << targetVals[j] << " ";
        outfile << "\n"
                << "Output: ";
        for (int j = 0; j < outputVals.size(); ++j)
            outfile << outputVals[j] << " ";
        outfile << "\n";
    }

    outfile.close();
}

void readTrainingData()
{
    ifstream infile;
    infile.open("training-set/training_input.dat");

    infile >> n >> rowCount >> colCount;
    n = NUM_TRAINING_TEST;

    for (int i = 0; i < n; ++i)
    {
        infile >> label[i];
        for (int x = 0; x < rowCount; ++x)
            for (int y = 0; y < colCount; ++y)
                infile >> image[i][x][y];
    }
    infile.close();

    n = NUM_TRAINING_TEST - 105;
};
