// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LAYERS_OPENCV_GRAY_INPUT_LAYER
#define _NEXURALNET_DNN_LAYERS_OPENCV_GRAY_INPUT_LAYER

#include "tensor.h"
#include "data_types.h"
#include "data_parser.h"

#include <opencv2\core\core.hpp>

namespace nexural {
	class OpenCVGrayImageLayer {
	public:
		OpenCVGrayImageLayer() {

		}

		~OpenCVGrayImageLayer() {

		}

		void Init(const LayerParams &layerParams) {
			_layerParams = layerParams;
			long nr = parser::ParseLong(_layerParams, "input_height");
			long nc = parser::ParseLong(_layerParams, "input_width");
			_inputShape.Resize(1, 1, nr, nc);
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		void LoadData(const cv::Mat& sourceImage) {
			if (sourceImage.channels() != 1) {
				throw std::runtime_error("Gray image must have 1 channel! Error details: " + std::to_string(__LINE__) + " " + __FUNCTION__);
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
