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

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "network.h"

int main(int argc, char* argv[]) {
	try {
		nexural::Tensor inputData, trainingData, targetData, inputImageData;
		boost::filesystem::path full_path(boost::filesystem::initial_path<boost::filesystem::path>());
		full_path = boost::filesystem::system_complete(boost::filesystem::path(argv[0]));
		std::string configFilePath = full_path.parent_path().string() + "\\network.json";
		//cv::Mat sourceImage = cv::imread(full_path.parent_path().parent_path().parent_path().parent_path().string() + "\\TestingData\\cat_3.png");
		// nexural::DataToTensorConverter::Convert(sourceImage, inputImageData);

		std::string trainingDataPath = full_path.parent_path().parent_path().parent_path().parent_path().string() + "\\TrainingData\\xor\\trainingData.txt";
		std::string targetDataPath = full_path.parent_path().parent_path().parent_path().parent_path().string() + "\\TrainingData\\xor\\targetData.txt";
		std::string inputDataPath = full_path.parent_path().parent_path().parent_path().parent_path().string() + "\\TestingData\\xor_inputData.txt";

		nexural::DataReader::ReadTensorFromFile(trainingDataPath, trainingData);
		nexural::DataReader::ReadTensorFromFile(targetDataPath, targetData);
		nexural::DataReader::ReadTensorFromFile(inputDataPath, inputData);

		nexural::Network net(configFilePath);
		nexural::NetworkTrainer netTrainer;
		netTrainer.Train(net, trainingData, targetData);
		//net.Run(inputImageData);
	}
	catch (std::exception stdEx) {
		std::cout << stdEx.what() << std::endl;
	}
	catch (...) {
		std::cout << "Something unexpected happened while running the network!" << std::endl;
	}
	return 0;
}