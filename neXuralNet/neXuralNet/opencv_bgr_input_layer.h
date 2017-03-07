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
#include "data_types.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_OPENCV_BGR_INPUT_LAYER
#define _NEXURALNET_DNN_LAYERS_OPENCV_BGR_INPUT_LAYER

namespace nexural {
	class OpenCVBGRImageLayer {
	public:
		OpenCVBGRImageLayer() {

		}

		~OpenCVBGRImageLayer() {

		}

		void Init(const LayerParams &layerParams) {
			_layerParams = layerParams;
			long nr = parser::ParseLong(_layerParams, "input_height");
			long nc = parser::ParseLong(_layerParams, "input_width");
			_inputShape.Resize(1, 3, nr, nc);
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		void LoadData(const cv::Mat& sourceImage) {
			if (sourceImage.channels() != 3) {
				throw std::runtime_error("BGR image must have 3 channels! Error details: " + std::to_string(__LINE__) + " " + __FUNCTION__);
			}

			// TODO: Add shape comparison between _outputData and soureImage

			for (long numSamples = 0; numSamples < _outputData.GetNumSamples(); numSamples++)
			{
				for (long nr = 0; nr < _outputData.GetNR(); nr++)
				{
					for (long nc = 0; nc < _outputData.GetNC(); nc++)
					{
						cv::Vec3b intensity = sourceImage.at<cv::Vec3b>(nr, nc);
						for (long k = 0; k < _outputData.GetK(); k++) {
							uchar col = intensity.val[k];
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = col;
						}
					}
				}
			}

		}

		Tensor* GetOutput() {
			return &_outputData;
		}

		LayerShape GetOutputShape() {
			return _outputShape;
		}

	private:
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
		LayerParams _layerParams;
	};
}
#endif
