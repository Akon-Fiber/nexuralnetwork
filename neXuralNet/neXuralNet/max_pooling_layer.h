// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LAYERS_MAX_POOLING_LAYER
#define _NEXURALNET_DNN_LAYERS_MAX_POOLING_LAYER

#include "data_types.h"
#include "computational_base_layer.h"
#include "data_parser.h"
#include <iostream>

namespace nexural {

	class MaxPoolingLayer : public ComputationalBaseLayer {
	public:
		MaxPoolingLayer(LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {
			_kernel_width = parser::ParseLong(_layerParams, "kernel_width");
			_kernel_height = parser::ParseLong(_layerParams, "kernel_height");
		}

		~MaxPoolingLayer() {

		}

		virtual void Setup(LayerShape& prevLayerShape) {
			long outNR = prevLayerShape.GetNR() / _kernel_height + (prevLayerShape.GetNR() % _kernel_height == 0 ? 0 : 1);
			long outNC = prevLayerShape.GetNC() / _kernel_width + (prevLayerShape.GetNC() % _kernel_width == 0 ? 0 : 1);

			_outputData.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), outNR, outNC);
			_maxIndexes.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), prevLayerShape.GetNR(), prevLayerShape.GetNC());
		}

		virtual void FeedForward(const Tensor& inputData) {


			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < inputData.GetK(); k++)
				{
					long outNR = 0;
					for (long nr = 0; nr < inputData.GetNR(); nr += _kernel_height)
					{
						long outNC = 0;
						for (long nc = 0; nc < inputData.GetNC(); nc += _kernel_width)
						{
							long khLimit = _kernel_height - (nr == (inputData.GetNR() - (inputData.GetNR() % _kernel_height)) ? _kernel_height - inputData.GetNR() % _kernel_height : 0);
							long kwLimit = _kernel_width - (nc == (inputData.GetNC() - (inputData.GetNC() % _kernel_width)) ? _kernel_width - inputData.GetNC() % _kernel_width : 0);
							
							long maxIndex = 0;
							float maxValue = std::numeric_limits<float>::min();

							for (long kh = 0; kh < khLimit; kh++)
							{
								for (long kw = 0; kw < kwLimit; kw++)
								{
									long currentIndex = (((numSamples * inputData.GetK()) + k) * inputData.GetNR() + (nr + kh)) * inputData.GetNC() + (nc + kw);
									_maxIndexes[currentIndex] = 0;
									float value = inputData[currentIndex];
									if (value >= maxValue) {
										maxValue = value;
										maxIndex = currentIndex;
									}
								}
							}

							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + outNR) * _outputData.GetNC() + outNC] = maxValue;
							_maxIndexes[maxIndex] = 1;
							outNC++;
						}
						outNR++;
					}
				}
			}

		}

		virtual void BackPropagate(const Tensor& layerErrors) {

		}

		virtual void Update() {

		}


	private:
		long _kernel_width;
		long _kernel_height;
		Tensor _maxIndexes;
	};

}
#endif
