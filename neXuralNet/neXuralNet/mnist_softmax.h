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

void Test_MNIST_Softmax() {
	nexural::Tensor inputData, trainingData, targetData;

	std::string networkConfigPath = "d:\\RESEARCH\\neXuralNetwork\\data\\mnist_softmax\\network.json";
	std::string trainerConfigPath = "d:\\RESEARCH\\neXuralNetwork\\data\\mnist_softmax\\trainer.json";

	nexural::DataReader::ReadMNISTData("D:\\RESEARCH\\Datasets\\MNIST\\train-images.idx3-ubyte", trainingData);
	nexural::DataReader::ReadMNISTLabels("D:\\RESEARCH\\Datasets\\MNIST\\train-labels.idx1-ubyte", targetData);

	nexural::Network net(networkConfigPath);
	//nexural::NetworkTrainer netTrainer(trainerConfigPath);
	//netTrainer.Train(net, trainingData, targetData);
}