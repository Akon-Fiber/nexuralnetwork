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
#include "tensor.h"
#include "data_parser.h"

#ifndef _NEXURALNET_UTILITY_DATA_TO_TENSOR_CONVERTER
#define _NEXURALNET_UTILITY_DATA_TO_TENSOR_CONVERTER

namespace nexural {
	class DataToTensorConverter {
	public:
		static void Convert(const cv::Mat& sourceImage, Tensor& outputData) {
			outputData.Resize(1, sourceImage.channels(), sourceImage.rows, sourceImage.cols);

			for (long numSamples = 0; numSamples < outputData.GetNumSamples(); numSamples++)
			{
				for (long nr = 0; nr < outputData.GetNR(); nr++)
				{
					for (long nc = 0; nc < outputData.GetNC(); nc++)
					{
						cv::Vec3b intensity = sourceImage.at<cv::Vec3b>(nr, nc);
						for (long k = 0; k < outputData.GetK(); k++) {
							uchar col = intensity.val[k];
							outputData[(((numSamples * outputData.GetK()) + k) * outputData.GetNR() + nr) * outputData.GetNC() + nc] = col;
						}
					}
				}
			}
		}
	};
}
#endif
