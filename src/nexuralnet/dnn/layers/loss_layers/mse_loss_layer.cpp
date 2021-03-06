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

#include "mse_loss_layer.h"

namespace nexural {
	MSELossLayer::MSELossLayer(const Params& layerParams) : LossBaseLayer(layerParams) {
		_resultType = NetworkResultType::REGRESSION;
	}

	MSELossLayer::~MSELossLayer() {

	}

	void MSELossLayer::Setup(const LayerShape& prevLayerShape) {
		if (prevLayerShape.GetK() != 1 || prevLayerShape.GetNR() != 1 || prevLayerShape.GetNC() != 1) {
			throw std::runtime_error("MSE layer error: Previous layer don't have a correct shape!");
		}
		_inputShape.Resize(prevLayerShape.GetNumSamples(), 1, 1, 1);
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
	}

	void MSELossLayer::FeedForward(const Tensor& inputData, const NetworkState) {
		for (long index = 0; index < inputData.Size(); index++) {
			_outputData[index] = inputData[index];
		}
	}

	void MSELossLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
	}

	void MSELossLayer::CalculateError(const Tensor& targetData) {
		if (_outputData.GetShape() != targetData.GetShape()) {
			throw std::runtime_error("MSE layer error: The output and target data should have the same size!");
		}

		long n = _outputData.GetNumSamples();
		float_n factor = float_n(2) / n;

		for (long numSamples = 0; numSamples < n; numSamples++)
		{
				float_n error = factor * (_outputData[numSamples ] - targetData[numSamples]);
				_layerErrors[numSamples] = error;
		}
	}

	void MSELossLayer::CalculateTrainingMetrics(const Tensor& targetData, Tensor& confusionMatrix) {
		if (_outputData.GetShape() != targetData.GetShape()) {
			throw std::runtime_error("MSE layer error: The output and target data should have the same size!");
		}
		
		float_n totalError = 0;
		long n = _outputData.GetNumSamples();
		for (long numSamples = 0; numSamples < n; numSamples++)
		{
			totalError += ((_outputData[numSamples] - targetData[numSamples]) *
					(_outputData[numSamples] - targetData[numSamples])) / n;
		}

		_totalError = totalError;
	}

	void MSELossLayer::SetResult() {
		_netResult.result.clear();
		for (long numSamples = 0; numSamples < _outputData.GetNumSamples(); numSamples++) {
			_netResult.result.push_back(_outputData[numSamples]);
		}
	}

	DNNBaseResult* MSELossLayer::GetResult() {
		return &_netResult;
	}

	const std::string MSELossLayer::GetResultJSON() {
		std::string resultJSON = u8"{ \"result_type\": \"" + helper::NetworkResultTypeToString(_resultType) + "\"";
		for (long numSamples = 0; numSamples < _netResult.result.size(); numSamples++) {
			resultJSON += ",\"result_" + std::to_string(numSamples) + "\" : \"" + std::to_string(_netResult.result[numSamples]) + "\"";
		}
		resultJSON += "}";
		return resultJSON;
	}
}
