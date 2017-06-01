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
#include "converter.h"
#include <opencv2/highgui.hpp>

#ifndef _NEXURALNET_TOOLS_DATA_WRITER_H
#define _NEXURALNET_TOOLS_DATA_WRITER_H

namespace nexural {
	namespace tools {
		class DataWriter {
		public:
			// TODO: This implementation is not safe
			static void WriteTensorImages(const std::string& outputFolder, Tensor& tensor) {
				std::vector<cv::Mat> images;
				converter::CvtTensorToVecOfMat(tensor, images);
				
				for (size_t idx = 0; idx < images.size(); idx++) {
					std::string imageName = outputFolder + "//image" + std::to_string(idx) + ".jpg";
					cv::imwrite(imageName, images[idx]);
				}
			}
		};
	}
}
#endif
