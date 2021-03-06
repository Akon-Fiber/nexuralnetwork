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

#include "fully_connected_layer.h"
#include "../../utility/params_parser.h"

namespace nexural {
	FullyConnectedLayer::FullyConnectedLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {
		_numOutputNeurons = parser::ParseLong(_layerParams, "neurons");
		_hasWeights = true;
		_hasBiases = true;
	}

	FullyConnectedLayer::~FullyConnectedLayer() {

	}

	void FullyConnectedLayer::Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
		_inputShape.Resize(prevLayerShape);
		_outputShape.Resize(_inputShape.GetNumSamples(), 1, 1, _numOutputNeurons);
		_outputData.Resize(_outputShape);
		_weights.Resize(1, 1, _numOutputNeurons, (_inputShape.GetK() * _inputShape.GetNR() * _inputShape.GetNC()));
		_biases.Resize(1, 1, 1, _numOutputNeurons);

		float_n weightRange = (float_n)(std::sqrt(1. / (double)_inputShape.Size()));
		float_n biasRange = (float_n)(std::sqrt(1. / (double)_biases.Size()));
		_weights.FillRandom(weightRange);
		_biases.Fill(biasRange);
		_layerID = "fully_connected_layer" + std::to_string(layerIndex);
	}

	void FullyConnectedLayer::FeedForward(const Tensor& inputData, const NetworkState networkState) {
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
							long inputIdx = (((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc;
							float_n inputValue = inputData[inputIdx];

							long weightsIdx = (n * _weights.GetNC() + ((k * inputData.GetNR() + nr) * inputData.GetNC() + nc));
							float_n weightValue = _weights[weightsIdx];
							neuronCalculatedValue += (inputValue * weightValue);
						}
					}
				}
				_outputData[numSamples * _outputData.GetNC() + n] = neuronCalculatedValue + _biases[n];
			}
		}
	}

	void FullyConnectedLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
		_dWeights.Resize(_weights.GetShape());
		_dBiases.Resize(_biases.GetShape());
	}

	void FullyConnectedLayer::BackPropagate(const Tensor& prevLayerErrors) {
		_dWeights.Fill(0.0);
		_dBiases.Fill(0.0);

		// Calculate gradient wrt. weights: (_internalInputData * prevLayerErrors)
		// Calculate gradient wrt. biases: (prevLayerErrors * 1)
		for (long numSamples = 0; numSamples < _internalInputData.GetNumSamples(); numSamples++)
		{
			for (long n = 0; n < _numOutputNeurons; n++)
			{
				float_n error = prevLayerErrors[numSamples * prevLayerErrors.GetNC() + n];
				_dBiases[n] += error;

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
			for (long k = 0; k < _layerErrors.GetK(); k++)
			{
				for (long nr = 0; nr < _layerErrors.GetNR(); nr++)
				{
					for (long nc = 0; nc < _layerErrors.GetNC(); nc++)
					{
						float_n layerErrorsValue = 0;

						for (long n = 0; n < _numOutputNeurons; n++)
						{
							float_n error = prevLayerErrors[numSamples * prevLayerErrors.GetNC() + n];
							long indx = (n * _weights.GetNC() + ((k * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc));
							float_n value = _weights[indx];
							layerErrorsValue += value * error;
						}

						_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc] = layerErrorsValue;
					}
				}
			}
		}
	}

	void FullyConnectedLayer::Serialize(Serializer& serializer) {
		serializer.AddParentNode(_layerID);
		serializer.SerializeTensor(_weights, _layerID, "weights");
		serializer.SerializeTensor(_biases, _layerID, "biases");
	}

	void FullyConnectedLayer::Deserialize(Serializer& serializer) {
		serializer.DeserializeTensor(_weights, _layerID, "weights");
		serializer.DeserializeTensor(_biases, _layerID, "biases");
	}
}
