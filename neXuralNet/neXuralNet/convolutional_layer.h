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
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER
#define _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER

namespace nexural {
	class ConvolutionalLayer : public ComputationalBaseLayer {
	public:
		ConvolutionalLayer(const LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {
			_numOfFilters = parser::ParseLong(_layerParams, "num_of_filters");
			_kernelWidth = parser::ParseLong(_layerParams, "kernel_width");
			_kernelHeight = parser::ParseLong(_layerParams, "kernel_height");
			_paddingWidth = parser::ParseLong(_layerParams, "padding_width");
			_paddingHeight = parser::ParseLong(_layerParams, "padding_height");
			_strideWidth = parser::ParseLong(_layerParams, "stride_width");
			_strideHeight = parser::ParseLong(_layerParams, "stride_height");
			_hasWeights = true;
			_hasBiases = true;
		}

		~ConvolutionalLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape, const int layerIndex) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(_inputShape.GetNumSamples(), _numOfFilters,
				(((_inputShape.GetNR() - _kernelHeight - (2 * _paddingHeight)) / _strideHeight)  + 1), 
				(((_inputShape.GetNC() - _kernelWidth - (2 * _paddingWidth)) / _strideWidth) + 1));
			_outputData.Resize(_outputShape);
			_weights.Resize(_numOfFilters, _inputShape.GetK(), _kernelHeight, _kernelWidth);
			_biases.Resize(1, 1, 1, _numOfFilters);
			_weights.FillRandom();
			_biases.FillRandom();
			_layerID = "convolutional_layer" + std::to_string(layerIndex);
		}

		virtual void FeedForward(const Tensor& inputData) {
			_internalInputData.ShareTensor(inputData);
			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++) {
				for (long numFilters = 0; numFilters < _weights.GetNumSamples(); numFilters++) {
					long nro = 0;
					for (long nr = 0; nr < inputData.GetNR() - _kernelHeight + 1; nr += _strideHeight) {
						long nco = 0;
						for (long nc = 0; nc < inputData.GetNC() - _kernelWidth + 1; nc += _strideWidth) {
							float value = 0;
							for (long i = 0; i < _kernelHeight; i++) {
								for (long j = 0; j < _kernelWidth; j++) {
									for (long k = 0; k < inputData.GetK(); k++) {
										value += inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + (nr + i)) * inputData.GetNC() + (nc + j)] *
										_weights[(((numFilters * _weights.GetK()) + k) * _weights.GetNR() + i) * _weights.GetNC() + j];
									}
								}
							}
							value += _biases[numFilters];
							_outputData[(((numSamples * _outputData.GetK()) + numFilters) * _outputData.GetNR() + nro) * _outputData.GetNC() + nco] = value;
							nco++;
						}
						nro++;
					}
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
			for (long errorNumSamples = 0; errorNumSamples < prevLayerErrors.GetNumSamples(); errorNumSamples++) {
				for (long k = 0; k < _internalInputData.GetK(); k++) {
					long nro = 0;
					for (long nr = 0; nr < _internalInputData.GetNR() - prevLayerErrors.GetNR() + 1; nr += _strideHeight) {
						long nco = 0;
						for (long nc = 0; nc < _internalInputData.GetNC() - prevLayerErrors.GetNC() + 1; nc += _strideWidth) {
							float error = 0;
							for (long i = 0; i < prevLayerErrors.GetNR(); i++) {
								for (long j = 0; j < prevLayerErrors.GetNC(); j++) {
									for (long inputNumSamples = 0; inputNumSamples < _internalInputData.GetNumSamples(); inputNumSamples++) {
										error += _internalInputData[(((inputNumSamples * _internalInputData.GetK()) + k) * _internalInputData.GetNR() + (nr + i)) * _internalInputData.GetNC() + (nc + j)] *
											prevLayerErrors[(((errorNumSamples * prevLayerErrors.GetK()) + k) * prevLayerErrors.GetNR() + i) * prevLayerErrors.GetNC() + j];
									}
								}
							}
							_dWeights[(((errorNumSamples * _dWeights.GetK()) + k) * _dWeights.GetNR() + nro) * _dWeights.GetNC() + nco] = error;
							nco++;
						}
						nro++;
					}
				}
			}

			// Calculate gradient wrt. biases: (1 * prevLayerErrors)
			long gbTotal = prevLayerErrors.GetK() * prevLayerErrors.GetNR() * prevLayerErrors.GetNC();
			for (long errorNumSamples = 0; errorNumSamples < prevLayerErrors.GetNumSamples(); errorNumSamples++) {
				for (long index = 0; index < gbTotal; index++) {
					prevLayerErrors[(errorNumSamples * gbTotal) + index];
				}
			}

			// Calculate gradient wrt. input: (_weights * prevLayerErrors)

		}

		virtual void Serialize(std::string& data) {
			DataSerializer::AddParentNode(_layerID, data);
			DataSerializer::SerializeTensor(_weights, _layerID, "weights", data);
			DataSerializer::SerializeTensor(_biases, _layerID, "biases", data);
		}

		virtual void Deserialize(const std::string& data) {
			DataSerializer::DeserializeTensor(_weights, _layerID, "weights", data);
			DataSerializer::DeserializeTensor(_biases, _layerID, "biases", data);
		}

	private:
		void AddPadding(const Tensor& input, Tensor& output, int padding) {

		}

	private:
		Tensor _internalInputData;
		long _numOfFilters;
		long _kernelWidth;
		long _kernelHeight;
		long _paddingWidth;
		long _paddingHeight;
		long _strideWidth;
		long _strideHeight;
	};
}
#endif
