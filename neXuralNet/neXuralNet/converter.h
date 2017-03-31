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

#ifndef _NEXURALNET_UTILITY_CONVERTER_H
#define _NEXURALNET_UTILITY_CONVERTER_H

namespace nexural {
	namespace converter {
		static void ConvertToTensor(const cv::Mat& sourceImage, Tensor& outputData) {
			outputData.Resize(1, sourceImage.channels(), sourceImage.rows, sourceImage.cols);

			for (long nr = 0; nr < outputData.GetNR(); nr++)
			{
				for (long nc = 0; nc < outputData.GetNC(); nc++)
				{
					cv::Vec3b intensity = sourceImage.at<cv::Vec3b>(nr, nc);
					for (long k = 0; k < outputData.GetK(); k++) {
						float col = (float)intensity.val[k];
						outputData[(k * outputData.GetNR() + nr) * outputData.GetNC() + nc] = col;
					}
				}
			}
		}

		static void ConvertToTensor(const std::vector<cv::Mat>& sourceImages, Tensor& outputData) {
			if (sourceImages.size() == 0) {
				return;
			}

			long numSamples = static_cast<long>(sourceImages.size());
			long channels = static_cast<long>(sourceImages[0].channels());
			long nr = sourceImages[0].rows;
			long nc = sourceImages[0].cols;

			outputData.Resize(numSamples, channels, nr, nc);

			for (long imageNumber = 0; imageNumber < numSamples; imageNumber++) {
				int currentChannels = sourceImages[0].channels();
				long currentNr = sourceImages[0].rows;
				long currentNc = sourceImages[0].cols;

				if (currentChannels != channels || currentNr != nr || currentNc != nc) {
					throw std::runtime_error("All the samples inside a tensor shoud have the same size!");
				}

				for (long nr = 0; nr < outputData.GetNR(); nr++)
				{
					for (long nc = 0; nc < outputData.GetNC(); nc++)
					{
						cv::Vec3b intensity = sourceImages[imageNumber].at<cv::Vec3b>(nr, nc);
						for (long k = 0; k < outputData.GetK(); k++) {
							float col = (float)intensity.val[k];
							outputData[(k * outputData.GetNR() + nr) * outputData.GetNC() + nc] = col;
						}
					}
				}
			}
		}
	}
}
#endif
