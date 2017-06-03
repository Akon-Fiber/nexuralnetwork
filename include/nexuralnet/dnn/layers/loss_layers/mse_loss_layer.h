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
#include "regression_result.h"

#ifndef _NEXURALNET_DNN_LAYERS_MSE_LOSS_LAYER
#define _NEXURALNET_DNN_LAYERS_MSE_LOSS_LAYER

namespace nexural {
	class MSELossLayer : public LossBaseLayer {
	public:
		MSELossLayer(const Params& layerParams) : LossBaseLayer(layerParams) {

		}

		~MSELossLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape) {
			if (prevLayerShape.GetK() != 1 || prevLayerShape.GetNR() != 1) {
				throw std::runtime_error("MSE layer error: Previous layer don't have a correct shape!");
			}
			_inputShape.Resize(prevLayerShape.GetNumSamples(), 1, 1, prevLayerShape.GetNC());
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		virtual void FeedForward(const Tensor& inputData) {
			for (long index = 0; index < inputData.Size(); index++) {
				_outputData[index] = inputData[index];
			}
		}

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
		}

		virtual void CalculateError(const Tensor& targetData) {
			if (_outputData.GetShape() != targetData.GetShape()) {
				throw std::runtime_error("MSE layer error: The output and target data should have the same size!");
			}
			
			long n = _outputData.GetNumSamples();
			float_n factor = float_n(2) / n;
			
			for (long numSamples = 0; numSamples < n; numSamples++)
			{
				for (long nc = 0; nc < _outputData.GetNC(); nc++)
				{
					float_n error = factor * (_outputData[numSamples * _outputData.GetNC() + nc] -
						targetData[numSamples * _outputData.GetNC() + nc]);
					
					_layerErrors[numSamples * _outputData.GetNC() + nc] = error;
				}
			}
		}

		virtual void CalculateTotalError(const Tensor& targetData) {
			if (_outputData.GetShape() != targetData.GetShape()) {
				throw std::runtime_error("MSE layer error: The output and target data should have the same size!");
			}

			long n = _outputData.GetNumSamples();
			_totalError = 0;
			for (long numSamples = 0; numSamples < n; numSamples++)
			{
				for (long nc = 0; nc < _outputData.GetNC(); nc++)
				{
					long idx = numSamples * _outputData.GetNC() + nc;
					_totalError += ((_outputData[numSamples * _outputData.GetNC() + nc] -
						targetData[numSamples * _outputData.GetNC() + nc]) *
						(_outputData[numSamples * _outputData.GetNC() + nc] -
							targetData[numSamples * _outputData.GetNC() + nc])) / n;
				}
			}
		}

		virtual void SetResult() {
			// TODO: Support multiple samples
			if (_outputData.Size() == 1) {
				_netResult.result = _outputData[0];
			}
			else {
				throw std::runtime_error("Can't set mse result values for tensors with multiple samples!");
			}
		}

		virtual DNNBaseResult* GetResult() {
			return &_netResult;
		}

		virtual const std::string GetResultJSON() {
			std::string resultJSON = u8"{ \
				\"result_type\": \"regression\", \
				\"result\" : \"" + std::to_string(_netResult.result) + "\" \
		}";
			return resultJSON;
		}

	private:
		RegressionResult _netResult;
	};
}
#endif
