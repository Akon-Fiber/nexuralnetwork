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
#include <fstream>
#include <algorithm>
#include <random>
#include "network.h"

enum class TestDataType {
	XOR = 0,
	AND = 1,
	XOR_SOFTMAX = 2,
	AND_SOFTMAX = 3
};

void GenerateTestData(TestDataType testDataType) {
	std::map <int, std::string> xorTraining{ { 0, "1 0" }, { 1, "0 1" }, { 2, "1 1" }, { 3, "0 0" } };
	std::map <int, std::string> xorTarget{ { 0, "1" }, { 1, "1" }, { 2, "0" }, { 3, "0" } };
	std::map <int, std::string> andTraining{ { 0, "1 0" }, { 1, "0 1" }, { 2, "1 1" }, { 3, "0 0" } };
	std::map <int, std::string> andTarget{ { 0, "0" }, { 1, "0" }, { 2, "1" }, { 3, "0" } };
	std::map <int, std::string> xorSoftmaxTarget{ { 0, "0 1" },{ 1, "0 1" },{ 2, "1 0" },{ 3, "1 0" } };
	std::map <int, std::string> andSoftmaxTarget{ { 0, "1 0" },{ 1, "1 0" },{ 2, "0 1" },{ 3, "1 0" } };

	std::ofstream fileTrainingData, fileTargetData;
	fileTrainingData.open("trainingData.txt");
	fileTargetData.open("targetData.txt");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 3);

	std::cout << "Number of pairs:" << std::endl;
	int iterations = 0;
	std::cin >> iterations;

	fileTrainingData << iterations << " 1 1 2";
	if(testDataType == TestDataType::XOR_SOFTMAX || testDataType == TestDataType::AND_SOFTMAX) {
		fileTargetData << iterations << " 1 1 2";
	}
	else 
	{
		fileTargetData << iterations << " 1 1 1";
	}

	switch (testDataType)
	{
	case TestDataType::AND:
		for (int i = 0; i < iterations; i++) {
			int randomIndex = dis(gen);
			fileTrainingData << "\n" << andTraining[randomIndex];
			fileTargetData << "\n" << andTarget[randomIndex];
		}
		break;
	case TestDataType::XOR:
		for (int i = 0; i < iterations; i++) {
			int randomIndex = dis(gen);
			fileTrainingData << "\n" << xorTraining[randomIndex];
			fileTargetData << "\n" << xorTarget[randomIndex];
		}
		break;
	case TestDataType::XOR_SOFTMAX:
		for (int i = 0; i < iterations; i++) {
			int randomIndex = dis(gen);
			fileTrainingData << "\n" << xorTraining[randomIndex];
			fileTargetData << "\n" << xorSoftmaxTarget[randomIndex];
		}
		break;
	case TestDataType::AND_SOFTMAX:
		for (int i = 0; i < iterations; i++) {
			int randomIndex = dis(gen);
			fileTrainingData << "\n" << xorTraining[randomIndex];
			fileTargetData << "\n" << andSoftmaxTarget[randomIndex];
		}
		break;
	default:
		break;
	}
	fileTrainingData.close();
	fileTargetData.close();
}