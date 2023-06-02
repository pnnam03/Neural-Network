#include <bits/stdc++.h>
using namespace std;

/*************** struct Connection ***************/
struct Connection
{
	double weight;
	double deltaWeight;
};

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
		int sign = rand() % 2 == 0 ? 1 : -1;
		return 1.0 * sign * rand() / double(RAND_MAX);
	}
	double activationFunction(double x);
	double activationFunctionDerivative(double x);

	vector<Layer> layerList;
	static double eta;
	static double alpha;
};

double Net::eta = 0.15;	 //
double Net::alpha = 0.5; // momentum

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

		// outputValue of Bias neuron is always 1
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

int rep = 0;
double sumDif = 0;
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
		outputLayer[i].gradient = 2 * (outputLayer[i].outputVal - targetVals[i]) *
								  activationFunctionDerivative(outputLayer[i].weightedInput);

		dif += (outputLayer[i].outputVal - targetVals[i]) *
			   (outputLayer[i].outputVal - targetVals[i]);
	}
	// rep++;
	// if (rep > 1500 && dif > 0.01)
	// 	cout << dif << "\n";
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
		curLayer[i].gradient = sumGradient * activationFunctionDerivative(curLayer[i].weightedInput);
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
	return tanh(x);
}

double Net::activationFunctionDerivative(double x)
{
	// derivative of sigmoid
	// double sigmoid = activationFunction(x);
	// return sigmoid * (1 - sigmoid);

	// derivative of tanh
	double val = tanh(x);
	return 1.0 - val * val;
}

/***************** Main program here ********************/

int main()
{

	vector<int> topology;
	topology.push_back(2);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet = Net(topology);

	// training
	vector<double> inputVals;
	vector<double> outputVals;
	vector<double> targetVals;

	inputVals.push_back(0), inputVals.push_back(0);
	targetVals.push_back(0);

	for (int j = 0; j < 2000; ++j)
	{
		int x = rand() % 2;
		int y = rand() % 2;
		int z = x ^ y;
		inputVals[0] = x, inputVals[1] = y;
		targetVals[0] = z;
		myNet.feedForward(inputVals);
		myNet.backPropagation(targetVals);
	}

	myNet.saveNet(topology);

	// testing
	ofstream outfile;
	outfile.open("out.txt");

	int cntCorrect = 0;
	for (int j = 1; j <= 100; ++j)
	{
		int x = rand() % 2;
		int y = rand() % 2;
		int z = x ^ y;

		inputVals[0] = x, inputVals[1] = y;
		targetVals[0] = z;

		myNet.feedForward(inputVals);
		myNet.getResult(outputVals);

		outfile << "Test #" << j << "\n";
		outfile << "Input: " << x << " " << y << "\n";
		outfile << "Output: ";
		for (int i = 0; i < outputVals.size(); ++i)
			outfile << outputVals[i] << " ";
		outfile << "\n";
		outfile << "Target: ";
		for (int i = 0; i < outputVals.size(); ++i)
			outfile << targetVals[i] << " ";
		outfile << "\n";

		if (abs(outputVals[0] - targetVals[0]) < 0.2)
			cntCorrect++;
	}

	outfile << cntCorrect << "%";

	outfile.close();
}
