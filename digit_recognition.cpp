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
int cnt = 0;
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
  Net()
  {
    cerr << "empty constructor";
  }
  Net(vector<int> &topology);
  void feedForward(vector<double> &inputVals);
  void backPropagation(vector<double> &targetVals);
  void updateWeight();
  int getResult();
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
  static double ACTIVATION_RATE;
  static double LEARNING_RATE;
  static double MINIMUM_LEARNING_RATE;
};

double Net::eta = 0.15;  //
double Net::alpha = 0.5; // momentum
double Net::ACTIVATION_RATE = 0.07;
double Net::LEARNING_RATE = 1.0;
double Net::MINIMUM_LEARNING_RATE = 0.005;
const int REP = 6000000;
//#define NEWNET

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
    }
  }
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

void Net::saveNet(vector<int> &topology)
{
  ofstream outfile;
  outfile.open("net.dat");

  outfile << topology.size() << " ";
  for (int i = 0; i < topology.size(); ++i)
    outfile << topology[i] << " ";

  for (int i = 0; i < layerList.size() - 1; ++i)
  {
    for (int j = 0; j < layerList[i].size(); ++j)
    {
      for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
        outfile << layerList[i][j].outputWeight[k].weight << " ";
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

//    dif += (outputLayer[i].outputVal - targetVals[i]) *
//           (outputLayer[i].outputVal - targetVals[i]);
  }
  rep++;
//  sum += dif;
  int result = getResult();
  if (targetVals[result] > 0)
    cnt++;
  if (rep % 1000 == 0)
  {
    cerr << 100.0 * cnt / rep << "%\n";
    //cerr<<sum /rep << "\n";
    //rep = 0;
    //cnt = 0;
    // sum = 0;
    Net::LEARNING_RATE -= 1.0/(REP / 1000);
    Net::LEARNING_RATE = max(Net::LEARNING_RATE, MINIMUM_LEARNING_RATE);
  }
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

      con.weight -= con.deltaWeight* LEARNING_RATE;
    }
}

double Net::activationFunction(double x)
{
  // sigmoid
  // return 1.0 / (1 + exp(-x));

  // tanh
  // return tanh(x);

  // ReLU
  return max(x, 0.0) * ACTIVATION_RATE;
}

double Net::activationDerivative(double x)
{
  // derivative of sigmoid
  // double sigmoid = activationFunction(x);
  // return sigmoid * (1 - sigmoid);

  // derivative of tanh
  // double val = tanh(x);
  // return 1.0 - val * val;

  // derivative of ReLU
  return 1.0 * (x > 0) * ACTIVATION_RATE;
}

/***************** Main program here ********************/

const int NUM_TRAINING_TEST = 60000;

int label[NUM_TRAINING_TEST + 1];
int image[NUM_TRAINING_TEST + 1][30][30];
int image1[NUM_TRAINING_TEST + 1][30][30];

void readTrainData();
void getNet(vector<int> &topolpgy, Net &myNet);

int n, rowCount, colCount;

int main()
{
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  srand(time(NULL));
  // freopen("bin","w",stdout);
  //  get the training data
  readTrainData();

  Net myNet = Net();
  vector<int> topology;

  // initialize new net
#ifdef NEWNET
  topology.push_back(rowCount * colCount);
  topology.push_back(20);
  topology.push_back(20);
  topology.push_back(10);
  myNet = Net(topology);
#endif // NEWNET

  // get Net from file
#ifndef NEWNET
  getNet(topology, myNet);
#endif // NEWNET

  // training
  vector<double> inputVals;
  vector<double> targetVals;
  vector<double> outputVals;

  for (int x = 0; x < rowCount; ++x)
    for (int y = 0; y < colCount; ++y)
      inputVals.push_back(0);
  for (int j = 0; j < 10; ++j)
    targetVals.push_back(0);

  for (int i = 0; i < REP; ++i)
  {
    if (i%1000 == 0)
      cerr<<i<<" ";
    int id = rand() * rand() % NUM_TRAINING_TEST;

    for (int x = 0; x < rowCount; ++x)
      for (int y = 0; y < colCount; ++y)
      {
        inputVals[x * colCount + y] = 1.0 * image[id][x][y] / 255;
      }

    // insert random noise
      int rep = rand() % 40;
      for (int j = 0; j < rep; ++j)
      {
        int pos = rand() % 784;
        int x = rand() % 256;
        inputVals[pos] = 1.0 * x / 255;
      }

    //
    for (int j = 0; j < targetVals.size(); ++j)
      targetVals[j] = 0;
    targetVals[label[id]] = 1;

    myNet.feedForward(inputVals);
    myNet.backPropagation(targetVals);
  }

  myNet.saveNet(topology);

  // testing

  int num_test = 10000;
  int correct = 0;
  ofstream outfile;
  outfile.open("out.txt");
  for (int i = 0; i < num_test; ++i)
  {
    int id = rand() * rand() % NUM_TRAINING_TEST;
    for (int x = 0; x < rowCount; ++x)
      for (int y = 0; y < colCount; ++y)
        inputVals[x * colCount + y] = 1.0 * image[id][x][y] / 255;

    myNet.feedForward(inputVals);
    int result = myNet.getResult();

    // outfile << "Test #" << i << "\n";
    // outfile << "Target: "<<label[id]<<"\n";
    // outfile << "Output: "<<result<<"\n";
    correct += result == label[id];
  }

  outfile << "Accuracy: " << 100.0 * correct / num_test << "%";
  outfile.close();
}

void readTrainData()
{
  ifstream infile;
  infile.open("training-set/training_input.dat");

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

void getNet(vector<int> &topology, Net &newNet)
{
  ifstream infile;
  infile.open("net.dat");
  topology.clear();
  int n;
  infile >> n;
  for (int i = 0; i < n; ++i)
  {
    int x;
    infile >> x;
    topology.push_back(x);
  }

  newNet = Net(topology);
  vector<Layer> &layerList = newNet.layerList;
  for (int i = 0; i < layerList.size() - 1; ++i)
  {
    for (int j = 0; j < layerList[i].size(); ++j)
    {
      for (int k = 0; k < layerList[i + 1].size() - 1; ++k)
        infile >> layerList[i][j].outputWeight[k].weight;
    }
  }

  infile.close();
}
