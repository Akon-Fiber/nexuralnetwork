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

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\mat.hpp>

#include <iostream>
#include <stdlib.h>
#include <conio.h>

#include "and_gate_with_tanh.h"
#include "and_gate_with_relu.h"
#include "xor_gate_with_relu.h"
#include "xor_gate_with_relu_softmax.h"
#include "and_gate_with_relu_softmax.h"
#include "mnist_softmax.h"
#include "generate_test_data.h"

#include "tensor.h"
#include "data_serializer.h"

void Menu() {
	std::cout << "--------------------------MENU--------------------------" << std::endl;
	std::cout << "| Available options:" << std::endl;
	std::cout << "| 0 - EXIT" << std::endl;
	std::cout << "| 1 - Random data for AND gate with Softmax" << std::endl;
	std::cout << "| 1 - Random data for XOR gate with Softmax" << std::endl;
	std::cout << "| 3 - Random data for AND gate" << std::endl;
	std::cout << "| 4 - Random data for XOR gate" << std::endl;
	std::cout << "| 5 - AND gate with TanH" << std::endl;
	std::cout << "| 6 - AND gate with RELU" << std::endl;
	std::cout << "| 7 - XOR gate with RELU" << std::endl;
	std::cout << "| 8 - AND gate with RELU and Softmax" << std::endl;
	std::cout << "| 9 - XOR gate with RELU and Softmax" << std::endl;
	std::cout << "| 10 - MNIST" << std::endl;
	std::cout << "| 11 - Experminetal" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl << std::endl;
}

void AddPadding(const nexural::Tensor& input, nexural::Tensor& output, long paddingWidth, long paddingHeight) {
	long newNR = input.GetNR() + 2 * paddingHeight;
	long newNC = input.GetNC() + 2 * paddingWidth;
	output.Resize(input.GetNumSamples(), input.GetK(), newNR, newNC);

	// Top and bottom bording
	for (long i = 0; i < newNC; i++) {
		output[i] = 0;
		output[i + output.GetNC() * (output.GetNR() - 1)] = 0;
	}

	// Left and right bording
	for (long j = 0; j < newNC; j++) {
		output[j * output.GetNC()] = 0;
		output[j * output.GetNC() + output.GetNC() - 1] = 0;
	}

	// Copy the data from input to output
	for (long numSamples = 0; numSamples < input.GetNumSamples(); numSamples++) {
		for (long k = 0; k < input.GetK(); k++) {
			long i = 0;
			for (long nr = paddingHeight; nr < newNR - paddingHeight; nr++) {
				long j = 0;
				for (long nc = paddingWidth; nc < newNC - paddingWidth; nc++) {
					output[(((numSamples * output.GetK()) + k) * output.GetNR() + nr) * output.GetNC() + nc] =
						input[(((numSamples * input.GetK()) + k) * input.GetNR() + i) * input.GetNC() + j];
					j++;
				}
				i++;
			}
		}
	}
}

void Experimental() {
	nexural::Tensor inputData, test;
	cv::Mat image = cv::imread("d:\\RESEARCH\\neXuralNetwork\\data\\images\\cat_3.png", CV_LOAD_IMAGE_COLOR);
	nexural::Utils::ConvertToTensor(image, inputData);

	AddPadding(inputData, test, 2, 2);

	std::string networkConfigPath = "d:\\RESEARCH\\neXuralNetwork\\neXuralNet\\ConfigFiles\\network.json";
	nexural::Network net(networkConfigPath);
	net.Run(inputData);
}

void DoTests(const int option) {
	switch (option) {
	case 1:
		GenerateTestData(TestDataType::AND_SOFTMAX);
		break;
	case 2:
		GenerateTestData(TestDataType::XOR_SOFTMAX);
		break;
	case 3:
		GenerateTestData(TestDataType::AND);
		break;
	case 4:
		GenerateTestData(TestDataType::XOR);
		break;
	case 5:
		Test_AND_Gate_With_TanH();
		break;
	case 6:
		Test_AND_Gate_With_RELU();
		break;
	case 7:
		Test_XOR_Gate_With_RELU();
		break;
	case 8:
		Test_AND_Gate_With_RELU_Softmax();
		break;
	case 9:
		Test_XOR_Gate_With_RELU_Softmax();
		break;
	case 10:
		Test_MNIST_Softmax();
		break;
	case 11:
		Experimental();
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[]) {
	try {
		int option = 1;
		while (option != 0) {
			Menu();
			std::cout << "Chose an option:" << std::endl;
			std::cin >> option;
			DoTests(option);
			std::cout << "Press any key to continue..." << std::endl;
			_getch();
			system("cls");
		}
	}
	catch (std::exception stdEx) {
		std::cout << stdEx.what() << std::endl;
	}
	catch (...) {
		std::cout << "Something unexpected happened while running the network!" << std::endl;
	}
	return 0;
}
