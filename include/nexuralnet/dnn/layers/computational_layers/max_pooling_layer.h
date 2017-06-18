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

#include "computational_base_layer.h"

#ifndef _NEXURALNET_DNN_LAYERS_MAX_POOLING_LAYER
#define _NEXURALNET_DNN_LAYERS_MAX_POOLING_LAYER

namespace nexural {
	class MaxPoolingLayer : public ComputationalBaseLayer {
	public:
		MaxPoolingLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {
			_kernel_width = parser::ParseLong(_layerParams, "kernel_width");
			_kernel_height = parser::ParseLong(_layerParams, "kernel_height");
		}

		~MaxPoolingLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
			if ((prevLayerShape.GetNR() % _kernel_height != 0) || (prevLayerShape.GetNC() % _kernel_width != 0)) {
				throw std::runtime_error("Max pooling layer error: Cannot apply max pooling to the input layer!");
			}
			
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), 
				(prevLayerShape.GetNR() / _kernel_height), (prevLayerShape.GetNC() / _kernel_width));

			_outputData.Resize(_outputShape);
			_maxIndexes.Resize(prevLayerShape);
			_layerID = "max_pooling_layer" + std::to_string(layerIndex);
		}

		virtual void FeedForward(const Tensor& inputData, const FeedForwardType feedForwardType = FeedForwardType::RUN) {
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
							long maxIndex = 0;
							float_n maxValue = std::numeric_limits<float_n>::min();

							for (long kh = 0; kh < _kernel_height; kh++)
							{
								for (long kw = 0; kw < _kernel_width; kw++)
								{
									long currentIndex = (((numSamples * inputData.GetK()) + k) * inputData.GetNR() + (nr + kh)) * inputData.GetNC() + (nc + kw);
									_maxIndexes[currentIndex] = 0;
									float_n value = inputData[currentIndex];
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

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
		}

		virtual void BackPropagate(const Tensor& prevLayerErrors) {
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
							float_n error = prevLayerErrors[(((numSamples * prevLayerErrors.GetK()) + k) * prevLayerErrors.GetNR() + outNR) * prevLayerErrors.GetNC() + outNC];

							for (long kh = 0; kh < _kernel_height; kh++)
							{
								for (long kw = 0; kw < _kernel_width; kw++)
								{
									if (_maxIndexes[(((numSamples * _maxIndexes.GetK()) + k) * _maxIndexes.GetNR() + (nr + kh)) * _maxIndexes.GetNC() + (nc + kw)] == 1) {
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

	private:
		long _kernel_width;
		long _kernel_height;
		Tensor _maxIndexes;
	};
}
#endif
