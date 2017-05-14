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

#include "stdafx.h"
#include <nexuralnet\tools.h>
#include <opencv2\imgcodecs.hpp>

using namespace nexural;

void Test_MNIST_Softmax(const std::string& dataFolderPath) {
	Tensor inputData, trainingData, targetData;
	Tensor testingData, testingTargetData;
	cv::Mat image;

	std::string exampleRoot = dataFolderPath + "\\mnist_softmax\\";
	std::string networkConfigPath = exampleRoot + "network.json";
	std::string trainerConfigPath = exampleRoot + "trainer.json";
	std::string trainingDataPath = exampleRoot + "train-images.idx3-ubyte";
	std::string targetDataPath = exampleRoot + "train-labels.idx1-ubyte";
	std::string testingDataPath = exampleRoot + "t10k-images.idx3-ubyte";
	std::string testingTargetDataPath = exampleRoot + "t10k-labels.idx1-ubyte";
	std::string testDataPath = exampleRoot + "test_images\\";

	int option = 0, numOfSamples = 0;
	std::cout << "1 - Train and test" << std::endl;
	std::cout << "2 - Test a pretrained network" << std::endl;
	std::cin >> option;

	Network net(networkConfigPath);

	if (option == 1) {
		std::cout << "Num of samples from dataset:" << std::endl;
		std::cin >> numOfSamples;

		std::cout << "Reading the training dataset..." << std::endl;
		tools::DataReader::ReadMNISTData(trainingDataPath, trainingData, numOfSamples);
		std::cout << "Reading the labels for the training dataset..." << std::endl << std::endl;
		tools::DataReader::ReadMNISTLabels(targetDataPath, targetData, numOfSamples);

		NetworkTrainer netTrainer(networkConfigPath, trainerConfigPath);
		netTrainer.Train(trainingData, targetData);
		netTrainer.Serialize(exampleRoot + "mnist.json");
	}

	net.Deserialize(exampleRoot + "mnist.json");

	std::cout << "Test the trained network: " << std::endl;


	std::cout << "Reading the testing dataset..." << std::endl;
	tools::DataReader::ReadMNISTData(testingDataPath, testingData, 50);
	std::cout << "Reading the labels for the testing dataset..." << std::endl << std::endl;
	tools::DataReader::ReadMNISTLabels(testingTargetDataPath, testingTargetData, 50);

	Tensor currentTestingData, currentTestingLabel;
	for (long numSamples = 0; numSamples < testingData.GetNumSamples(); numSamples++) {
		currentTestingData.GetBatch(testingData, numSamples);
		currentTestingLabel.GetBatch(testingTargetData, numSamples);
		std::cout << "Target:" << std::endl;
		currentTestingLabel.PrintToConsole();
		net.Run(currentTestingData);
		std::cout << std::endl;
	}

	image = cv::imread(testDataPath + "image0.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 7" << std::endl;
	net.Run(inputData);
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image1.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 2" << std::endl;
	net.Run(inputData);
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image2.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 1" << std::endl;
	net.Run(inputData);
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image3.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 0" << std::endl;
	net.Run(inputData);
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image4.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 4" << std::endl;
	net.Run(inputData);
	std::cout << std::endl;
}