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
		MaxPoolingLayer(const LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {
			_kernel_width = parser::ParseLong(_layerParams, "kernel_width");
			_kernel_height = parser::ParseLong(_layerParams, "kernel_height");
		}

		~MaxPoolingLayer() {

		}

		virtual void Setup(LayerShape& prevLayerShape) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), 
				(prevLayerShape.GetNR() / _kernel_height + (prevLayerShape.GetNR() % _kernel_height == 0 ? 0 : 1)),
				(prevLayerShape.GetNC() / _kernel_width + (prevLayerShape.GetNC() % _kernel_width == 0 ? 0 : 1)));

			_outputData.Resize(_outputShape);
			_maxIndexes.Resize(prevLayerShape);
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
			for (long numSamples = 0; numSamples < _layerErrors.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < _layerErrors.GetK(); k++)
				{
					long outNR = 0;
					for (long nr = 0; nr < _layerErrors.GetNR(); nr += _kernel_height)
					{
						long outNC = 0;
						for (long nc = 0; nc < _layerErrors.GetNC(); nc += _kernel_width)
						{
							long khLimit = _kernel_height - (nr == (_layerErrors.GetNR() - (_layerErrors.GetNR() % _kernel_height)) ? _kernel_height - _layerErrors.GetNR() % _kernel_height : 0);
							long kwLimit = _kernel_width - (nc == (_layerErrors.GetNC() - (_layerErrors.GetNC() % _kernel_width)) ? _kernel_width - _layerErrors.GetNC() % _kernel_width : 0);

							float error = layerErrors[(((numSamples * layerErrors.GetK()) + k) * layerErrors.GetNR() + outNR) * layerErrors.GetNC() + outNC];

							for (long kh = 0; kh < khLimit; kh++)
							{
								for (long kw = 0; kw < kwLimit; kw++)
								{
									if (_maxIndexes[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + (nr + kh)) * _layerErrors.GetNC() + (nc + kw)] == 1) {
										_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + (nr + kh)) * _layerErrors.GetNC() + (nc + kw)] = error;
									}
									else 
									{
										_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + (nr + kh)) * _layerErrors.GetNC() + (nc + kw)] = 0;
									}
								}
							}
							outNC++;
						}
						outNR++;
					}
				}
			}
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
