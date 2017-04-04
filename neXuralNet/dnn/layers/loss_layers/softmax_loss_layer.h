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

#ifndef _NEXURALNET_DNN_LAYERS_SOFTMAX_LOSS_LAYER
#define _NEXURALNET_DNN_LAYERS_SOFTMAX_LOSS_LAYER

namespace nexural {
	class SoftmaxLossLayer : public LossBaseLayer {
	public:
		SoftmaxLossLayer(const LayerParams& layerParams) : LossBaseLayer(layerParams) {

		}

		~SoftmaxLossLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape) {
			if (prevLayerShape.GetK() != 1 || prevLayerShape.GetNR() != 1) {
				throw std::runtime_error("Previous layer don't have a correct shape!");
			}
			_inputShape.Resize(prevLayerShape.GetNumSamples(), 1, 1, prevLayerShape.GetNC());
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		virtual void FeedForward(const Tensor& inputData) {
			Softmax(inputData, _outputData);
		}

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
		}

		virtual void CalculateError(const Tensor& targetData) {
			if (_outputData.GetShape() != targetData.GetShape()) {
				throw std::runtime_error("The output and target data should have the same size!");
			}

			Tensor aux, mask(targetData.GetShape());
			mask.Fill(0);
			Softmax(_outputData, aux);

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

		virtual void CalculateTotalError(const Tensor& targetData) {
			if (_outputData.GetShape() != targetData.GetShape()) {
				throw std::runtime_error("The output and target data should have the same size!");
			}

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
			_totalError = 0;
			for (long numSamples = 0; numSamples < totalOutputNumSamples; numSamples++)
			{
				long idx = numSamples * _outputData.GetNC() + indexes[numSamples];
				_totalError += -std::log(helper::clip(_outputData[idx], 1e-10f, 1.0f));
			}
		}

	private:
		void Softmax(const Tensor& inputData, Tensor& outputData) {
			long totalInputNumSamples = inputData.GetNumSamples();
			std::vector<float> max(totalInputNumSamples);
			_totalError = 0;
			for (long numSamples = 0; numSamples < totalInputNumSamples; numSamples++)
			{
				max[numSamples] = std::numeric_limits<float>::min();
				for (long nc = 0; nc < inputData.GetNC(); nc++)
				{
					long idx = numSamples * inputData.GetNC() + nc;
					if (max[numSamples] < inputData[idx]) {
						max[numSamples] = inputData[idx];
					}
				}
			}

			Tensor exp(inputData.GetShape());
			std::vector<float> sum(totalInputNumSamples);
			for (long numSamples = 0; numSamples < totalInputNumSamples; numSamples++)
			{
				sum[numSamples] = 0;
				for (long nc = 0; nc < inputData.GetNC(); nc++)
				{
					long idx = numSamples * inputData.GetNC() + nc;
					float expValue = std::exp(inputData[idx] - max[numSamples]); // numerically stable
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

	private:

	};
}
#endif
