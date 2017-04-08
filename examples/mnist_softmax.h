/* Copyright (C) 2016-2017 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include "../nexuralnet/dnn/network/network.h"

using namespace nexural;

void Test_MNIST_Softmax(const std::string& dataFolderPath) {
	Tensor inputData, trainingData, targetData;

	std::string networkConfigPath = dataFolderPath + "\\mnist_softmax\\network.json";
	std::string trainerConfigPath = dataFolderPath + "\\mnist_softmax\\trainer.json";
	std::string trainingDataPath = dataFolderPath + "\\mnist_softmax\\train-images.idx3-ubyte";
	std::string targetDataPath = dataFolderPath + "\\mnist_softmax\\train-labels.idx1-ubyte";

	std::cout << "Reading the training dataset..." << std::endl;
	tools::DataReader::ReadMNISTData(trainingDataPath, trainingData, 20000);
	std::cout << "Reading the labels for the training dataset..." << std::endl;
	tools::DataReader::ReadMNISTLabels(targetDataPath, targetData, 20000);

	std::cout << "Initialize the trainer..." << std::endl;
	Network net(networkConfigPath);
	NetworkTrainer netTrainer(trainerConfigPath);
	std::cout << "Starting the training process..." << std::endl;
	netTrainer.Train(net, trainingData, targetData);

	net.Serialize("D:\\mnist.json");
}