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

#include "softmax_loss_layer.h"

namespace nexural {
	SoftmaxLossLayer::SoftmaxLossLayer(const Params& layerParams) : LossBaseLayer(layerParams) {
		_resultType = NetworkResultType::MULTICLASS_CLASSIFICATION;
	}

	SoftmaxLossLayer::~SoftmaxLossLayer() {

	}

	void SoftmaxLossLayer::Setup(const LayerShape& prevLayerShape) {
		if (prevLayerShape.GetK() != 1 || prevLayerShape.GetNR() != 1) {
			throw std::runtime_error("Softmax layer error: Previous layer don't have a correct shape!");
		}
		_inputShape.Resize(prevLayerShape.GetNumSamples(), 1, 1, prevLayerShape.GetNC());
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
	}

	void SoftmaxLossLayer::FeedForward(const Tensor& inputData, const NetworkState) {
		Softmax(inputData, _outputData);
	}

	void SoftmaxLossLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
	}

	void SoftmaxLossLayer::CalculateError(const Tensor& targetData) {
		if (_outputData.GetShape() != targetData.GetShape()) {
			throw std::runtime_error("Softmax layer error: The output and target data should have the same size!");
		}

		Tensor aux, mask(targetData.GetShape());
		mask.Fill(0);

		long totalTargetNumSamples = targetData.GetNumSamples();
		for (long numSamples = 0; numSamples < totalTargetNumSamples; numSamples++)
		{
			for (long nc = 0; nc < targetData.GetNC(); nc++)
			{
				long idx = numSamples * targetData.GetNC() + nc;
				if (targetData[idx] == 1) {
					mask[idx] = 1;
					break;
				}
			}
		}

		for (long index = 0; index < _outputData.Size(); index++)
		{
			_layerErrors[index] = _outputData[index] - mask[index];
		}
	}

	void SoftmaxLossLayer::CalculateTrainingMetrics(const Tensor& targetData, Tensor& confusionMatrix) {
		if (_outputData.GetShape() != targetData.GetShape()) {
			throw std::runtime_error("Softmax layer error: The output and target data should have the same size!");
		}

		float_n totalError = 0;
		long totalTargetNumSamples = targetData.GetNumSamples();
		std::vector<int> indexes(totalTargetNumSamples);
		for (long numSamples = 0; numSamples < totalTargetNumSamples; numSamples++)
		{
			for (long nc = 0; nc < targetData.GetNC(); nc++)
			{
				long idx = numSamples * targetData.GetNC() + nc;
				if (targetData[idx] == 1) {
					indexes[numSamples] = idx;
					break;
				}
			}
		}

		long totalOutputNumSamples = _outputData.GetNumSamples();
		for (long numSamples = 0; numSamples < totalOutputNumSamples; numSamples++)
		{
			long idx = numSamples * _outputData.GetNC() + indexes[numSamples];
			totalError += -std::log(helper::clip(_outputData[idx], 1e-10, 1.0));
			long nr, nc;
			helper::BestClassClassification(_outputData, nc);
			helper::BestClassClassification(targetData, nr);
			confusionMatrix[nr * confusionMatrix.GetNC() + nc] += 1;
		}

		_totalError = totalError;
	}

	void SoftmaxLossLayer::SetResult() {
		_netResult.classesWithProbabilities.clear();
		_netResult.resultClass.clear();
		for (long numSamples = 0; numSamples < _outputData.GetNumSamples(); numSamples++) {
			long bestClass;
			Tensor aux;
			helper::BestClassClassification(_outputData, bestClass);
			aux.Clone(_outputData);
			_netResult.classesWithProbabilities.push_back(aux);
			_netResult.resultClass.push_back(bestClass);
		}
	}

	DNNBaseResult* SoftmaxLossLayer::GetResult() {
		return &_netResult;
	}

	const std::string SoftmaxLossLayer::GetResultJSON() {
		std::string resultJSON = u8"{ \"result_type\": \"" + helper::NetworkResultTypeToString(_resultType) + "\"";
		for (long numSamples = 0; numSamples < _netResult.resultClass.size(); numSamples++) {
			resultJSON += ",\"best_class_" + std::to_string(numSamples) + "\" : \"" + std::to_string(_netResult.resultClass[numSamples]) + "\"";
		}
		resultJSON += "}";
		return resultJSON;
	}

	void SoftmaxLossLayer::Softmax(const Tensor& inputData, Tensor& outputData) {
		long totalInputNumSamples = inputData.GetNumSamples();
		std::vector<float_n> max(totalInputNumSamples);
		for (long numSamples = 0; numSamples < totalInputNumSamples; numSamples++)
		{
			max[numSamples] = std::numeric_limits<float_n>::min();
			for (long nc = 0; nc < inputData.GetNC(); nc++)
			{
				long idx = numSamples * inputData.GetNC() + nc;
				if (max[numSamples] < inputData[idx]) {
					max[numSamples] = inputData[idx];
				}
			}
		}

		Tensor exp(inputData.GetShape());
		std::vector<float_n> sum(totalInputNumSamples);
		for (long numSamples = 0; numSamples < totalInputNumSamples; numSamples++)
		{
			sum[numSamples] = 0;
			for (long nc = 0; nc < inputData.GetNC(); nc++)
			{
				long idx = numSamples * inputData.GetNC() + nc;
				float_n expValue = std::exp(inputData[idx] - max[numSamples]); // numerically stable
				exp[idx] = expValue;
				sum[numSamples] += expValue;
			}
		}

		outputData.Resize(inputData.GetShape());
		long totalOutputNumSamples = outputData.GetNumSamples();
		for (long numSamples = 0; numSamples < totalOutputNumSamples; numSamples++)
		{
			for (long nc = 0; nc < outputData.GetNC(); nc++)
			{
				long idx = numSamples * outputData.GetNC() + nc;
				outputData[idx] = exp[idx] / sum[numSamples];
			}
		}
	}
}
