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

#ifndef _NEXURALNET_DNN_LAYERS_FULLY_CONNECTED_LAYER
#define _NEXURALNET_DNN_LAYERS_FULLY_CONNECTED_LAYER

namespace nexural {
	class FullyConnectedLayer : public ComputationalBaseLayer {
	public:
		FullyConnectedLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {
			_numOutputNeurons = parser::ParseLong(_layerParams, "neurons");
			_hasWeights = true;
			_hasBiases = true;
		}

		~FullyConnectedLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape, const int layerIndex) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(_inputShape.GetNumSamples(), 1, 1, _numOutputNeurons);
			_outputData.Resize(_outputShape);
			_weights.Resize(1, 1, _numOutputNeurons, (_inputShape.GetK() * _inputShape.GetNR() * _inputShape.GetNC()));
			_biases.Resize(1, 1, 1, _numOutputNeurons);

			float_n weightRange = (float_n)(std::sqrt(2. / (double)_inputShape.Size()));
			float_n biasRange = (float_n)(std::sqrt(2. / (double)_biases.Size()));
			_weights.FillRandom(weightRange);
			_biases.FillRandom(biasRange);
			_layerID = "fully_connected_layer" + std::to_string(layerIndex);
		}

		virtual void FeedForward(const Tensor& inputData) {
			_internalInputData.ShareTensor(inputData);
			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++)
			{
				for (long n = 0; n < _numOutputNeurons; n++)
				{
					float_n neuronCalculatedValue = 0;
					for (long k = 0; k < inputData.GetK(); k++)
					{
						for (long nr = 0; nr < inputData.GetNR(); nr++)
						{
							for (long nc = 0; nc < inputData.GetNC(); nc++)
							{
								float_n inputValue = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];
								
								long indx = (n * _weights.GetNC() + ((k * inputData.GetNR() + nr) * inputData.GetNC() + nc));
								float_n weightValue = _weights[indx];
								neuronCalculatedValue += (inputValue * weightValue);
							}
						}
					}
					_outputData[numSamples * inputData.GetNC() + n] = neuronCalculatedValue + _biases[n];
				}
			}
		}

		virtual void SetupLayerForTraining() {
			_layerErrors.Resize(_inputShape);
			_dWeights.Resize(_weights.GetShape());
			_dBiases.Resize(_biases.GetShape());
		}

		virtual void BackPropagate(const Tensor& prevLayerErrors) {
			_dWeights.Fill(0.0);
			_dBiases.Fill(0.0);

			// Calculate gradient wrt. weights: (_internalInputData * prevLayerErrors)
			// Calculate gradient wrt. biases: (prevLayerErrors * 1)
			for (long numSamples = 0; numSamples < _internalInputData.GetNumSamples(); numSamples++)
			{
				for (long n = 0; n < _numOutputNeurons; n++)
				{
					float_n error = prevLayerErrors[numSamples * prevLayerErrors.GetNC() + n];
					_dBiases[numSamples * prevLayerErrors.GetNC() + n] += error;

					for (long k = 0; k < _internalInputData.GetK(); k++)
					{
						for (long nr = 0; nr < _internalInputData.GetNR(); nr++)
						{
							for (long nc = 0; nc < _internalInputData.GetNC(); nc++)
							{
								float_n value = _internalInputData[(((numSamples * _internalInputData.GetK()) + k) * _internalInputData.GetNR() + nr) * _internalInputData.GetNC() + nc];
								long indx = (n * _dWeights.GetNC() + ((k * _internalInputData.GetNR() + nr) * _internalInputData.GetNC() + nc));
								_dWeights[indx] += (value * error);
							}
						}
					}
				}
			}

			// Calculate gradient wrt. input: (prevLayerErrors * _weights)
			for (long numSamples = 0; numSamples < _layerErrors.GetNumSamples(); numSamples++)
			{
				for (long n = 0; n < _numOutputNeurons; n++)
				{
					float_n error = prevLayerErrors[numSamples * prevLayerErrors.GetNC() + n];
					for (long k = 0; k < _layerErrors.GetK(); k++)
					{
						for (long nr = 0; nr < _layerErrors.GetNR(); nr++)
						{
							for (long nc = 0; nc < _layerErrors.GetNC(); nc++)
							{
								long indx = (n * _weights.GetNC() + ((k * _internalInputData.GetNR() + nr) * _internalInputData.GetNC() + nc));
								float_n value = _weights[indx];
								_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc] = value * error;
							}
						}
					}
				}
			}
		}

		virtual void Serialize(DataSerializer& serializer) {
			serializer.AddParentNode(_layerID);
			serializer.SerializeTensor(_weights, _layerID, "weights");
			serializer.SerializeTensor(_biases, _layerID, "biases");
		}

		virtual void Deserialize(DataSerializer& serializer) {
			serializer.DeserializeTensor(_weights, _layerID, "weights");
			serializer.DeserializeTensor(_biases, _layerID, "biases");
		}

	private:
		Tensor _internalInputData;
		long _numOutputNeurons;
	};
}
#endif
