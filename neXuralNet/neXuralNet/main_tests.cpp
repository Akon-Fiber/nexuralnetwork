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
#include <stdlib.h>
#include <conio.h>

#include "and_gate_with_tanh.h"
#include "and_gate_with_relu.h"
#include "xor_gate_with_relu.h"
#include "generate_test_data.h"

void Menu() {
	std::cout << "--------------------------MENU--------------------------" << std::endl;
	std::cout << "| Available options:" << std::endl;
	std::cout << "| 0 - EXIT" << std::endl;
	std::cout << "| 1 - AND gate with TanH" << std::endl;
	std::cout << "| 2 - AND gate with RELU" << std::endl;
	std::cout << "| 3 - Random data for AND gate" << std::endl;
	std::cout << "| 4 - Random data for XOR gate" << std::endl;
	std::cout << "| 5 - XOR gate with RELU" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl << std::endl;
}

void DoTests(const int option) {
	switch (option) {
	case 1:
		Test_AND_Gate_With_TanH();
		break;
	case 2:
		Test_AND_Gate_With_RELU();
		break;
	case 3:
		GenerateTestData(TestDataType::AND);
		break;
	case 4:
		GenerateTestData(TestDataType::XOR);
		break;
	case 5:
		Test_XOR_Gate_With_RELU();
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[]) {
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.setf(std::ios::showpoint);
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
