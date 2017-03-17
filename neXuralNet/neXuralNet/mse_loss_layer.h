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

#include "loss_base_layer.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_LOSS_LAYER
#define _NEXURALNET_DNN_LAYERS_LOSS_LAYER

namespace nexural {
	class MSELossLayer : public LossBaseLayer {
	public:
		MSELossLayer(const LayerParams& layerParams) : LossBaseLayer(layerParams) {

		}

		~MSELossLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape) {
			if (prevLayerShape.GetK() != 1 || prevLayerShape.GetNR != 1) {
				throw std::runtime_error("");
			}
			_inputShape.Resize(prevLayerShape.GetNumSamples(), 1, 1, prevLayerShape.GetNC());
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		virtual void FeedForward(const Tensor& inputData) {
			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < inputData.GetK(); k++)
				{
					for (long nr = 0; nr < inputData.GetNR(); nr++)
					{
						for (long nc = 0; nc < inputData.GetNC(); nc++)
						{
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];
						}
					}
				}
			}
		}

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
		}

		virtual void CalculateError(Tensor& targetData) {
			_layerErrors.Fill(0);
			float n = static_cast<float>(_outputData.GetNumSamples());
			float factor = float(2) / n;
			
			if(_outputData.GetShape() != targetData.GetShape()) {
				throw std::runtime_error("The output and target data should have the same size!");
			}

			for (long numSamples = 0; numSamples < _outputData.GetNumSamples(); numSamples++)
			{
				for (long nc = 0; nc < _outputData.GetNC(); nc++)
				{
					float error = targetData[numSamples * _outputData.GetNC() + nc] -
						_outputData[numSamples * _outputData.GetNC() + nc];
					error /= factor;
					
					//_layerErrors[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] += error;
				}


			}
		}

	private:

	};
}
#endif
