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

#include "selu_dropout_layer.h"
#include "../../utility/params_parser.h"

namespace nexural {
	SeluDropoutLayer::SeluDropoutLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {
		_threshold = parser::ParseFloat(_layerParams, "dropout_ratio");
		_alpha = parser::ParseFloat(_layerParams, "alpha");
		auto q = float_n(1) - _threshold;
		auto tmp = q * (float_n(1) + _alpha * _alpha * (float_n(1) - q));
		_a = std::pow(tmp, -0.5);
		_b = -_a * (1 - q) * _alpha;
	}

	SeluDropoutLayer::~SeluDropoutLayer() {

	}

	void SeluDropoutLayer::Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
		_inputShape.Resize(prevLayerShape);
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
		_dropoutIndexes.Resize(_outputShape);
		_layerID = "selu_dropout_layer" + std::to_string(layerIndex);
	}

	void SeluDropoutLayer::FeedForward(const Tensor& inputData, const NetworkState networkState) {
		if (networkState == NetworkState::TRAINING) {
			_dropoutIndexes.FillRandomBinomialDistribution(_threshold);

			for (long i = 0; i < inputData.Size(); i++)
			{
				float_n value = inputData[i];
				float_n drop = _dropoutIndexes[i];
				_outputData[i] = value * drop;
				_outputData[i] = _a * (drop == 1 ? value : _alpha) + _b;
			}
		}
		else {
			for (long i = 0; i < inputData.Size(); i++)
			{
				_outputData[i] = inputData[i];
			}
		}
	}

	void SeluDropoutLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
	}

	void SeluDropoutLayer::BackPropagate(const Tensor& prevLayerErrors) {
		for (long i = 0; i < _layerErrors.Size(); i++)
		{
			float_n error = prevLayerErrors[i];
			float_n drop = _dropoutIndexes[i];
			_layerErrors[i] = error * drop * _a;
		}
	}
}
