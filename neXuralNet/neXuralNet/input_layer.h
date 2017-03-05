// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef MAVNET_DNN_LAYERS_INPUT_LAYER
#define MAVNET_DNN_LAYERS_INPUT_LAYER

#include <map>
#include "tensor.h"
#include <opencv2\core\core.hpp>

#include <iostream>

namespace nexural {

	class InputOpenCVBGRImage {
	public:
		InputOpenCVBGRImage() {

		}

		~InputOpenCVBGRImage() {

		}

		void LoadImage(const cv::Mat& sourceImage) {
			if(sourceImage.channels() != 3){
				throw std::runtime_error("BGR image must have 3 channels! Error details: " + std::to_string(__LINE__) + " " + __FUNCTION__);
			}

			_outputData.Resize(1, sourceImage.channels(), sourceImage.rows, sourceImage.cols);

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

	private:
		Tensor _outputData;
	};

}
#endif
