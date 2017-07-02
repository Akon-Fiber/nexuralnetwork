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

#include "../dnn/data_types/tensor.h"

#ifndef NEXURALNET_TOOLS_DATA_READER_H
#define NEXURALNET_TOOLS_DATA_READER_H

namespace nexural {
	namespace tools {
		class DataReader {
		public:
			enum class ReadImageType {
				COLOR = 0,
				GRAY = 1
			};

			enum class ReadMNISTLabelsType {
				FOR_REGRESSION = 0,
				FOR_CLASSIFICATION = 1
			};

			static void ReadMNISTData(const std::string& filename, Tensor& tensor, long limit = 60000);
			static void ReadMNISTLabels(const std::string& filename, Tensor& tensor, long limit = 60000, ReadMNISTLabelsType labelsType = ReadMNISTLabelsType::FOR_CLASSIFICATION);
			static void ReadImagesFromDirectory(const std::string directoryPath, Tensor& tensor, ReadImageType imagesType = ReadImageType::COLOR);
			static void ReadTensorFromFile(const std::string filePath, Tensor& tensor);
			static void ReadATTData(const std::string& directoryPath, Tensor& training, Tensor& labels);

		private:
			static int ReverseInt(int i);
		};
	}
}
#endif
