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
#include "../dnn/network/network.h"

using namespace nexural;

void Test_XOR_Gate_With_RELU_Softmax(const std::string& dataFolderPath) {
	Tensor inputData, trainingData, targetData;

	std::string networkConfigPath = dataFolderPath + "\\xor_relu_softmax\\network.json";
	std::string trainerConfigPath = dataFolderPath + "\\xor_relu_softmax\\trainer.json";
	std::string trainingDataPath = dataFolderPath + "\\xor_relu_softmax\\trainingData.txt";
	std::string targetDataPath = dataFolderPath + "\\xor_relu_softmax\\targetData.txt";

	tools::DataReader::ReadTensorFromFile(trainingDataPath, trainingData);
	tools::DataReader::ReadTensorFromFile(targetDataPath, targetData);

	int option = 0;
	std::cout << "1 - Train and test" << std::endl;
	std::cout << "2 - Test apretrained network" << std::endl;
	std::cin >> option;

	Network net(networkConfigPath);

	if (option == 1) {
		NetworkTrainer netTrainer(trainerConfigPath);
		netTrainer.Train(net, trainingData, targetData);
		net.Serialize("D:\\netsave.json");
	}
	else {
		net.Deserialize("D:\\netsave.json");
	}

	std::cout << "Test the trained network: " << std::endl;
	inputData.Resize(1, 1, 1, 2);
	std::cout << "Input: 1 1 | Target: 1 0" << std::endl;
	inputData[0] = 1.0;
	inputData[1] = 1.0;
	net.Run(inputData);
	std::cout << "Input: 1 0 | Target: 0 1" << std::endl;
	inputData[0] = 1.0;
	inputData[1] = 0.0;
	net.Run(inputData);
	std::cout << "Input: 0 1 | Target: 0 1" << std::endl;
	inputData[0] = 0.0;
	inputData[1] = 1.0;
	net.Run(inputData);
	std::cout << "Input: 0 0 | Target: 1 0" << std::endl;
	inputData[0] = 0.0;
	inputData[1] = 0.0;
	net.Run(inputData);
}