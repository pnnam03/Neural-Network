#include <bits/stdc++.h>
using namespace std;

/*************** struct Connection ***************/
struct Connection
{
  double weight;
  double deltaWeight;
};

int rep = 0;
double sum = 0;
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
  Net();
  Net(vector<int> &topology);
  void feedForward(vector<double> &inputVals);
  int getResult();
  double activationFunction(double x);
  double activationDerivative(double x);
  vector<Layer> layerList;
  static double eta;
  static double alpha;
  static double ACTIVATION_RATE;
};

double Net::eta = 0.15; //
double Net::alpha = 0.5; // momentum
double Net::ACTIVATION_RATE = 0.07;
// #define NEWNET

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
    layerList.back().back().outputVal = 1;
  }

  for (int i = 0; i < topology.size() - 1; ++i)
  {
    for (int j = 0; j < layerList[i].size(); ++j)
    {
      Neuron &curNeuron = layerList[i][j];
      for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
      {
        curNeuron.outputWeight.push_back(Connection());
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
    }
  }
}

int Net::getResult()
{
  Layer &lastLayer = layerList.back();
  double maxVal = lastLayer[0].outputVal;
  int result = 0;
  for (int i = 0; i < lastLayer.size() - 1; ++i)
    if (lastLayer[i].outputVal > maxVal)
    {
      maxVal = lastLayer[i].outputVal;
      result = i;
    }

  return result;
}

double Net::activationFunction(double x)
{
  // sigmoid
  // return 1.0 / (1 + exp(-x));

  // tanh
  //    return tanh(x/15);

  // ReLU
  return max(x, 0.0) * ACTIVATION_RATE;
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
  return 1.0 * (x > 0) * ACTIVATION_RATE;
}

/***************** Main program here ********************/

const int NUM_TRAINING_TEST = 60000;

int label[NUM_TRAINING_TEST + 1];
int image[NUM_TRAINING_TEST + 1][30][30];
int image1[NUM_TRAINING_TEST + 1][30][30];

void readTestData();
Net getNet();

int n, rowCount, colCount;

int main()
{
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  srand(time(NULL));

  Net myNet = getNet();

  // training
  vector<double> inputVals;
  vector<double> targetVals;

  // testing
  readTestData();

  int num_test = n;
  int correct = 0;

  ofstream outfile;
  outfile.open("out.txt");
  vector<int> outputVals;

  num_test = 10000;
  for (int i = 0; i < num_test; ++i)
  {
    inputVals.clear();
    for (int x = 0; x < rowCount; ++x)
      for (int y = 0; y < colCount; ++y)
        inputVals.push_back(1.0 * image[i][x][y] / 255);

    myNet.feedForward(inputVals);
    int result = myNet.getResult();
    outputVals.push_back(result);
    // outfile << "Test #" << i << "\n";
    // outfile << "Target: "<<label[id]<<"\n";
    // outfile << "Output: "<<result<<"\n";
    correct += result == label[i];
  }

  outfile << "Accuracy: " << 100.0 * correct / num_test << "%\n";
  for (int i = 0; i < num_test; ++i)
    if (label[i] != outputVals[i])
    {
    outfile << "Test #" << i << "\n";
    outfile << "Target: " << label[i] << "\n";
    outfile << "Output: " << outputVals[i] << "\n";
  }
  outfile.close();
}

void readTestData()
{
  ifstream infile;
  infile.open("testing-set/test_input.dat");

  infile >> n >> rowCount >> colCount;

  for (int i = 0; i < n; ++i)
  {
    infile >> label[i];
    int dx = rowCount-1, dy = colCount-1;
    for (int x = 0; x < rowCount; ++x)
      for (int y = 0; y < colCount; ++y)
      {
        infile >> image1[i][x][y];
        image[i][x][y] = 0;
        if (image1[i][x][y])
        {
          dx = min(x,dx);
          dy = min(y,dy);
        }
      }

    // pull the image to (0,0)

    for (int x = dx; x < rowCount; ++x)
      for (int y = dy; y < colCount; ++y)
        image[i][x-dx][y-dy] = image1[i][x][y];
  }
  infile.close();
}

Net getNet()
{
  ifstream infile;
  infile.open("net.dat");
  vector<int> topology;
  topology.clear();
  int n;
  infile >> n;
  for (int i = 0; i < n; ++i)
  {
    int x;
    infile >> x;
    topology.push_back(x);
  }

  Net myNet = Net(topology);
  vector<Layer> &layerList = myNet.layerList;
  for (int i = 0; i < layerList.size() - 1; ++i)
  {
    for (int j = 0; j < layerList[i].size(); ++j)
    {
      for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
        infile >> layerList[i][j].outputWeight[k].weight;
    }
  }
  infile.close();

  return myNet;
}
