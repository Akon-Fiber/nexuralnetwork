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
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_AVERAGE_POOLING_LAYER
#define _NEXURALNET_DNN_LAYERS_AVERAGE_POOLING_LAYER

namespace nexural {
	class AveragePoolingLayer : public ComputationalBaseLayer {
	public:
		AveragePoolingLayer(const LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {
			_kernel_width = parser::ParseLong(_layerParams, "kernel_width");
			_kernel_height = parser::ParseLong(_layerParams, "kernel_height");
		}

		~AveragePoolingLayer() {
			
		}
		
		virtual void Setup(const LayerShape& prevLayerShape) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(),
				(prevLayerShape.GetNR() / _kernel_height + (prevLayerShape.GetNR() % _kernel_height == 0 ? 0 : 1)),
				(prevLayerShape.GetNC() / _kernel_width + (prevLayerShape.GetNC() % _kernel_width == 0 ? 0 : 1)));

			_outputData.Resize(_outputShape);
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
							
							float sum = 0;

							for (long kh = 0; kh < khLimit; kh++)
							{
								for (long kw = 0; kw < kwLimit; kw++)
								{
									float value = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + (nr + kh)) * inputData.GetNC() + (nc + kw)];
									sum += value;
								}
							}
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + outNR) * _outputData.GetNC() + outNC] = sum / (khLimit * kwLimit);
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
							long khLimit = _kernel_height - (nr == (_layerErrors.GetNR() - (_layerErrors.GetNR() % _kernel_height)) ? _kernel_height - _layerErrors.GetNR() % _kernel_height : 0);
							long kwLimit = _kernel_width - (nc == (_layerErrors.GetNC() - (_layerErrors.GetNC() % _kernel_width)) ? _kernel_width - _layerErrors.GetNC() % _kernel_width : 0);

							float error = prevLayerErrors[(((numSamples * prevLayerErrors.GetK()) + k) * prevLayerErrors.GetNR() + outNR) * prevLayerErrors.GetNC() + outNC] / (khLimit * kwLimit);

							for (long kh = 0; kh < khLimit; kh++)
							{
								for (long kw = 0; kw < kwLimit; kw++)
								{
									_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + (nr + kh)) * _layerErrors.GetNC() + (nc + kw)] = error;
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
	};
}
#endif
