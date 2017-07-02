/* Copyright (C) 2017 Alexandru-Valentin Musat (contact@nexuralsoftware.com)

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

#include "relu_layer.h"
#include "../../utility/params_parser.h"

namespace nexural {
	ReluLayer::ReluLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {

	}

	ReluLayer::~ReluLayer() {

	}

	void ReluLayer::Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
		_inputShape.Resize(prevLayerShape);
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
		_layerID = "relu_layer" + std::to_string(layerIndex);
	}

	void ReluLayer::FeedForward(const Tensor& inputData, const FeedForwardType feedForwardType) {
		_internalInputData.ShareTensor(inputData);
		for (long i = 0; i < inputData.Size(); i++)
		{
			float_n value = inputData[i];
			_outputData[i] = value < 0 ? (float_n)0.0 : value;
		}
	}

	void ReluLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
	}

	void ReluLayer::BackPropagate(const Tensor& prevLayerErrors) {
		for (long i = 0; i < _internalInputData.Size(); i++)
		{
			float_n error = prevLayerErrors[i];
			float_n value = _internalInputData[i];
			_layerErrors[i] = value <= 0 ? (float_n)0.0 : error;
		}
	}
}
