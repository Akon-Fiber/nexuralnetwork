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

#include <stdlib.h>
#include <nexuralnet/experimental.h>

#include "and_gate_with_tanh.h"
#include "and_gate_with_relu.h"
#include "xor_gate_with_relu.h"
#include "xor_gate_with_relu_softmax.h"
#include "and_gate_with_relu_softmax.h"
#include "mnist_softmax.h"
#include "att_softmax.h"

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
	std::cout << "| 11 - ATT with Softmax" << std::endl;
	std::cout << "| e - Experminetal" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl << std::endl;
}

void DoTests(const std::string& option, const std::string& dataFolderPath) {
	if (option == "0") {

	}
	else if (option == "1") {
		GenerateTestData(tools::TestDataType::AND_SOFTMAX);
	}
	else if (option == "2") {
		GenerateTestData(tools::TestDataType::XOR_SOFTMAX);
	}
	else if (option == "3") {
		GenerateTestData(tools::TestDataType::AND);
	}
	else if (option == "4") {
		GenerateTestData(tools::TestDataType::XOR);
	}
	else if (option == "5") {
		Test_AND_Gate_With_TanH(dataFolderPath);
	}
	else if (option == "6") {
		Test_AND_Gate_With_RELU(dataFolderPath);
	}
	else if (option == "7") {
		Test_XOR_Gate_With_RELU(dataFolderPath);
	}
	else if (option == "8") {
		Test_AND_Gate_With_RELU_Softmax(dataFolderPath);
	}
	else if (option == "9") {
		Test_XOR_Gate_With_RELU_Softmax(dataFolderPath);
	}
	else if (option == "10") {
		Test_MNIST_Softmax(dataFolderPath);
	}
	else if (option == "11") {
		Test_ATT_Softmax(dataFolderPath);
	}
	else if (option == "e") {
		nexural::experimental::TestDemo();
	} else {
		std::cout << "Option inserted isn't correct! Please, try again!" << std::endl;
	}
}

int main(int argc, char* argv[]) {
	std::string userInput;

	if (argc != 2) {
		std::cout << "Error: Firstly, specify the data folder!" << std::endl;
		std::cout << "Press any alphanumeric key and enter to exit..." << std::endl;
		std::cin >> userInput;
		return -1;
	}

	std::string dataFolderPath = argv[1];
	std::string option = "1";

	while (option != "0") {
		try {
			Menu();
			std::cout << "Chose an option:" << std::endl;
			std::cin >> option;
			DoTests(option, dataFolderPath);
		}
		catch (std::exception stdEx) {
			std::cout << stdEx.what() << std::endl;
		}
		catch (...) {
			std::cout << "Something unexpected happened while running the network!" << std::endl;
		}
		std::cout << "Press any alphanumeric key and enter to continue..." << std::endl;
		std::cin >> userInput;
		system("cls");
	}
	return 0;
}
