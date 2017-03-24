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

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <fstream>
#include <string>

#include "tensor.h"

#ifndef _NEXURALNET_UTILITY_DATA_READER_H
#define _NEXURALNET_UTILITY_DATA_READER_H

namespace nexural {
	class DataReader {
	public:
		enum class ReadImageType {
			COLOR,
			GRAY
		};
	
		static void ReadImagesFromDirectory(const std::string directoryPath, TensorCollection& tensorCollection, ReadImageType imagesType = ReadImageType::COLOR) {
			//auto readType = imagesType == ReadImageType::COLOR ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
			//std::vector<cv::String> fn;
			//cv::glob("/home/images/*.png", fn, false);

			//for (long i = 0; i < fn.size(); i++) {
			//	//cv::Mat image = cv::imread(fn[i], readType);
			//	//Tensor tensor;
			//	//DataToTensorConverter::Convert(image, tensor);
			//	//tensorCollection.push_back(tensor);
			//}
		}

		// TODO: Change it with a class that supports RAII
		static void ReadTensorFromFile(const std::string filePath, Tensor& tensor) {
			std::ifstream file(filePath, std::ios::binary);
			if (file.fail()) {
				throw std::runtime_error("Can't open file!");
			}

			long index = 0, numSamples, k, nr, nc;
			file >> numSamples;
			file >> k;
			file >> nr;
			file >> nc;
			tensor.Resize(numSamples, k, nr, nc);
			long size = tensor.Size() - 1;

			while (!file.eof())
			{
				float value;
				file >> value;
				tensor[index] = value;
				if (size == index) {
					break;
				}
				else {
					index++;
				}
			}
			file.close();
		}

	};
}
#endif
