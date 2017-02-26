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
			if(sourceImage.channels() == 3){
				//TODO throw exception
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

			for (int i = 0; i < _outputData.Size(); i++) {
				std::cout << _outputData[i] << std::endl;
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
