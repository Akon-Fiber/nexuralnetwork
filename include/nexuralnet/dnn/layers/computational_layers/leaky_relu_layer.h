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

#ifndef _NEXURALNET_DNN_LAYERS_LEAKY_RELU_LAYER
#define _NEXURALNET_DNN_LAYERS_LEAKY_RELU_LAYER

namespace nexural {
	class LeakyReluLayer : public ComputationalBaseLayer {
	public:
		LeakyReluLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {

		}

		~LeakyReluLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
			_layerID = "leaky_relu_layer" + std::to_string(layerIndex);
		}

		virtual void FeedForward(const Tensor& inputData) {
			_internalInputData.ShareTensor(inputData);
			for (long i = 0; i < inputData.Size(); i++)
			{
				float_n value = inputData[i];
				_outputData[i] = value < 0 ? (float_n)0.01 * value : value;
			}
		}

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
		}

		virtual void BackPropagate(const Tensor& prevLayerErrors) {
			for (long i = 0; i < _internalInputData.Size(); i++)
			{
				float_n error = prevLayerErrors[i];
				float_n value = _internalInputData[i];
				_layerErrors[i] = value < 0 ? (float_n)0.01 * error : error;
			}
		}

	private:
		Tensor _internalInputData;
	};
}
#endif
