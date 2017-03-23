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
#include "network.h"

void Test_AND_Gate_With_TanH() {
	nexural::Tensor inputData, trainingData, targetData;

	std::string networkConfigPath = "d:\\RESEARCH\\neXuralNetwork\\data\\and_tanh\\network.json";
	std::string trainingDataPath = "d:\\RESEARCH\\neXuralNetwork\\data\\and_tanh\\trainingData.txt";
	std::string targetDataPath = "d:\\RESEARCH\\neXuralNetwork\\data\\and_tanh\\targetData.txt";

	nexural::DataReader::ReadTensorFromFile(trainingDataPath, trainingData);
	nexural::DataReader::ReadTensorFromFile(targetDataPath, targetData);

	nexural::Network net(networkConfigPath);
	nexural::NetworkTrainer netTrainer;
	netTrainer.Train(net, trainingData, targetData);

	std::cout << "Test the trained network: " << std::endl;
	inputData.Resize(1, 1, 1, 2);
	std::cout << "Input: 1 1 | Target: 1" << std::endl;
	inputData[0] = 1.0;
	inputData[1] = 1.0;
	net.Run(inputData);
	std::cout << "Input: 1 0 | Traget: 0" << std::endl;
	inputData[0] = 1.0;
	inputData[1] = 0.0;
	net.Run(inputData);
	std::cout << "Input: 0 1 | Traget: 0" << std::endl;
	inputData[0] = 0.0;
	inputData[1] = 1.0;
	net.Run(inputData);
	std::cout << "Input: 0 0 | Traget: 0" << std::endl;
	inputData[0] = 0.0;
	inputData[1] = 0.0;
	net.Run(inputData);
}