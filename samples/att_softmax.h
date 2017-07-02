/* Copyright (C) 2017 Alexandru-Valentin Musat (contact@nexuralsoftware.com)

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
#include <opencv2/imgcodecs.hpp>

using namespace nexural;

void Test_ATT_Softmax(const std::string& dataFolderPath) {
	Tensor inputData, trainingData, targetData;
	Tensor testingData, testingTargetData;
	MultiClassClassificationResult* netResult;
	cv::Mat image;

	std::string exampleRoot = dataFolderPath + "\\att_softmax\\";
	std::string networkConfigPath = exampleRoot + "network.json";
	std::string trainerConfigPath = exampleRoot + "trainer.json";
	std::string testDataPath = exampleRoot + "test_images\\";
	std::string filtersImagesPath = exampleRoot + "filters_images\\";
	std::string trainerInfoDataPath = exampleRoot + "trainerInfo.json";
	std::string trainedDataFilePath = exampleRoot + "trainedData.json";
	std::string imagesFolderPath = exampleRoot + "images\\";

	int option = 0;
	std::cout << "1 - Train and test" << std::endl;
	std::cout << "2 - Test a pretrained network" << std::endl;
	std::cout << "3 - Save filters images" << std::endl;
	std::cin >> option;

	Network net(networkConfigPath);

	if (option == 1) {
		std::cout << "Reading the training dataset..." << std::endl;
		tools::DataReader::ReadATTData(imagesFolderPath, trainingData, targetData);

		NetworkTrainer netTrainer(networkConfigPath, trainerConfigPath);
		netTrainer.Train(trainingData, targetData, trainedDataFilePath, exampleRoot);
	}

	net.Deserialize(trainedDataFilePath);

	std::cout << std::endl << "Test the trained network: " << std::endl << std::endl;

	image = cv::imread(testDataPath + "image0.jpg", cv::IMREAD_GRAYSCALE);
	std::cout << "Target: 7" << std::endl;
	net.Run(image);
	netResult = dynamic_cast<MultiClassClassificationResult*>(net.GetResult());
	std::cout << "Result: " << netResult->resultClass << std::endl;
	std::cout << std::endl;

	if (option == 3) {
		net.SaveFiltersImages(filtersImagesPath);
		return;
	}

	image = cv::imread(testDataPath + "image1.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::CvtMatToTensor(image, inputData);
	std::cout << "Target: 2" << std::endl;
	net.Run(image);
	netResult = dynamic_cast<MultiClassClassificationResult*>(net.GetResult());
	std::cout << "Result: " << netResult->resultClass << std::endl;
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image2.jpg", cv::IMREAD_GRAYSCALE);
	std::cout << "Target: 1" << std::endl;
	net.Run(image);
	netResult = dynamic_cast<MultiClassClassificationResult*>(net.GetResult());
	std::cout << "Result: " << netResult->resultClass << std::endl;
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image3.jpg", cv::IMREAD_GRAYSCALE);
	std::cout << "Target: 0" << std::endl;
	net.Run(image);
	netResult = dynamic_cast<MultiClassClassificationResult*>(net.GetResult());
	std::cout << "Result: " << netResult->resultClass << std::endl;
	std::cout << std::endl;

	image = cv::imread(testDataPath + "image4.jpg", cv::IMREAD_GRAYSCALE);
	std::cout << "Target: 4" << std::endl;
	net.Run(image);
	netResult = dynamic_cast<MultiClassClassificationResult*>(net.GetResult());
	std::cout << "Result: " << netResult->resultClass << std::endl;
	std::cout << std::endl;
}