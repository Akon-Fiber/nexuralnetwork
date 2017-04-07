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

#include <stdlib.h>
#include <conio.h>

#include "and_gate_with_tanh.h"
#include "and_gate_with_relu.h"
#include "xor_gate_with_relu.h"
#include "xor_gate_with_relu_softmax.h"
#include "and_gate_with_relu_softmax.h"
#include "mnist_softmax.h"
#include "../nexuralnet/tools/generate_test_data.h"

void Menu() {
	std::cout << "--------------------------MENU--------------------------" << std::endl;
	std::cout << "| Available options:" << std::endl;
	std::cout << "| 0 - EXIT" << std::endl;
	std::cout << "| 1 - Random data for AND gate with Softmax" << std::endl;
	std::cout << "| 1 - Random data for XOR gate with Softmax" << std::endl;
	std::cout << "| 3 - Random data for AND gate" << std::endl;
	std::cout << "| 4 - Random data for XOR gate" << std::endl;
	std::cout << "| 5 - AND gate with TanH with MSE" << std::endl;
	std::cout << "| 6 - AND gate with RELU with MSE" << std::endl;
	std::cout << "| 7 - XOR gate with RELU with MSE" << std::endl;
	std::cout << "| 8 - AND gate with RELU and Softmax" << std::endl;
	std::cout << "| 9 - XOR gate with RELU and Softmax" << std::endl;
	std::cout << "| 10 - MNIST with Softmax" << std::endl;
	std::cout << "| 11 - Experminetal" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl << std::endl;
}

void Experimental(const std::string& dataFolderPath) {
	Tensor inputData;

	std::string networkConfigPath = dataFolderPath + "\\mnist_softmax\\network.json";
	Network net(networkConfigPath);
	net.Deserialize("D:\\mnist.json");

	cv::Mat image = cv::imread("D:\\RESEARCH\\nexural_tests\\5.jpg", cv::IMREAD_GRAYSCALE);
	nexural::converter::ConvertToTensor(image, inputData);
	net.Run(inputData);


	//long _strideHeight = 1, _strideWidth = 1;
	//Tensor convPrevLayerErrors;
	//convPrevLayerErrors.Resize(1, 2, 4, 4);
	//convPrevLayerErrors.Fill(std::vector<float_n>{0, 0, 0, 0, 0, 3, 4, 0, 0, 6, 8, 0, 0, 0, 0, 0,
	//											0, 0, 0, 0, 0, 5, 5, 0, 0, 2, 3, 0, 0, 0, 0, 0});
	//convPrevLayerErrors.OutputToConsole();

	//Tensor _weights;
	//_weights.Resize(2, 3, 2, 2);
	//_weights.Fill(std::vector<float_n>{2, 7, 5, 1, 9, 3, 2, 4, 6, 1, 4, 7, 8, 7, 1, 3, 5, 9, 3, 5, 4, 4, 6, 8});
	//_weights.OutputToConsole();

	//Tensor _layerErrors;
	//_layerErrors.Resize(1, 3, 3, 3);
	//_layerErrors.Fill(0.0);

	//for (long numSamples = 0; numSamples < convPrevLayerErrors.GetNumSamples(); numSamples++) {
	//	for (long k = 0; k < _weights.GetK(); k++) {
	//		long nro = 0;
	//		for (long nr = 0; nr < convPrevLayerErrors.GetNR() - _weights.GetNR() + 1; nr += _strideHeight) {
	//			long nco = 0;
	//			for (long nc = 0; nc < convPrevLayerErrors.GetNC() - _weights.GetNC() + 1; nc += _strideWidth) {
	//				float_n error = 0;
	//				for (long i = 0; i < _weights.GetNR(); i++) {
	//					for (long j = 0; j < _weights.GetNC(); j++) {
	//						for (long numOfFilters = 0; numOfFilters < convPrevLayerErrors.GetK(); numOfFilters++) {
	//							error += convPrevLayerErrors[(((numSamples * convPrevLayerErrors.GetK()) + numOfFilters) * convPrevLayerErrors.GetNR() + (nr + i)) * convPrevLayerErrors.GetNC() + (nc + j)] 
	//								* _weights[(((numOfFilters * _weights.GetK()) + k) * _weights.GetNR() + i) * _weights.GetNC() + j];
	//						}
	//					}
	//				}
	//				_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nro) * _layerErrors.GetNC() + nco] = error;
	//				nco++;
	//			}
	//			nro++;
	//		}
	//	}
	//}

	//_layerErrors.OutputToConsole();

}

void DoTests(const int option, const std::string& dataFolderPath) {
	switch (option) {
	case 1:
		GenerateTestData(tools::TestDataType::AND_SOFTMAX);
		break;
	case 2:
		GenerateTestData(tools::TestDataType::XOR_SOFTMAX);
		break;
	case 3:
		GenerateTestData(tools::TestDataType::AND);
		break;
	case 4:
		GenerateTestData(tools::TestDataType::XOR);
		break;
	case 5:
		Test_AND_Gate_With_TanH(dataFolderPath);
		break;
	case 6:
		Test_AND_Gate_With_RELU(dataFolderPath);
		break;
	case 7:
		Test_XOR_Gate_With_RELU(dataFolderPath);
		break;
	case 8:
		Test_AND_Gate_With_RELU_Softmax(dataFolderPath);
		break;
	case 9:
		Test_XOR_Gate_With_RELU_Softmax(dataFolderPath);
		break;
	case 10:
		Test_MNIST_Softmax(dataFolderPath);
		break;
	case 11:
		Experimental(dataFolderPath);
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[]) {
	try {
		if (argc != 2) {
			std::cout << "Error: Firstly, specify the data folder!" << std::endl;
			std::cout << "Press any key to continue..." << std::endl;
			_getch();
			return -1;
		}
		std::string dataFolderPath = argv[1];
		int option = 1;
		while (option != 0) {
			Menu();
			std::cout << "Chose an option:" << std::endl;
			std::cin >> option;
			DoTests(option, dataFolderPath);
			std::cout << "Press any key to continue..." << std::endl;
			_getch();
			system("cls");
		}
	}
	catch (std::exception stdEx) {
		std::cout << stdEx.what() << std::endl;
		_getch();
	}
	catch (...) {
		std::cout << "Something unexpected happened while running the network!" << std::endl;
	}
	return 0;
}
