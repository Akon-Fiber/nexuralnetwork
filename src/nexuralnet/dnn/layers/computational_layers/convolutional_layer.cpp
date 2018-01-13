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

#include "convolutional_layer.h"
#include "../../utility/params_parser.h"

namespace nexural {
	ConvolutionalLayer::ConvolutionalLayer(const Params &layerParams) : ComputationalBaseLayer(layerParams) {
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

	ConvolutionalLayer::~ConvolutionalLayer() {

	}

	void ConvolutionalLayer::Setup(const LayerShape& prevLayerShape, const size_t layerIndex) {
		_inputShape.Resize(prevLayerShape);
		_outputShape.Resize(_inputShape.GetNumSamples(), _numOfFilters,
			(((_inputShape.GetNR() - _kernelHeight - (2 * _paddingHeight)) / _strideHeight) + 1),
			(((_inputShape.GetNC() - _kernelWidth - (2 * _paddingWidth)) / _strideWidth) + 1));
		_outputData.Resize(_outputShape);
		_weights.Resize(_numOfFilters, _inputShape.GetK(), _kernelHeight, _kernelWidth);
		_biases.Resize(1, 1, 1, _numOfFilters);

		float_n weightRange = (float_n)(std::sqrt(1. / (double)_inputShape.Size()));
		float_n biasRange = (float_n)(std::sqrt(1. / (double)_biases.Size()));
		_weights.FillRandom(weightRange);
		_biases.Fill(biasRange);
		_layerID = "convolutional_layer" + std::to_string(layerIndex);
	}

	void ConvolutionalLayer::FeedForward(const Tensor& inputData, const NetworkState networkState) {
		_internalInputData.ShareTensor(inputData);
		for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++) {
			for (long numFilters = 0; numFilters < _weights.GetNumSamples(); numFilters++) {
				long nro = 0;
				for (long nr = 0; nr < inputData.GetNR() - _weights.GetNR() + 1; nr += _strideHeight) {
					long nco = 0;
					for (long nc = 0; nc < inputData.GetNC() - _weights.GetNC() + 1; nc += _strideWidth) {
						float_n value = 0;
						for (long i = 0; i < _weights.GetNR(); i++) {
							for (long j = 0; j < _weights.GetNC(); j++) {
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

	void ConvolutionalLayer::SetupLayerForTraining() {
		_layerErrors.Resize(_inputShape);
		_dWeights.Resize(_weights.GetShape());
		_dBiases.Resize(_biases.GetShape());
	}

	void ConvolutionalLayer::BackPropagate(const Tensor& prevLayerErrors) {
		// Calculate gradient wrt. weights: (_internalInputData * prevLayerErrors)
		for (long numOfFilters = 0; numOfFilters < prevLayerErrors.GetK(); numOfFilters++) {
			for (long k = 0; k < _internalInputData.GetK(); k++) {
				long nro = 0;
				for (long nr = 0; nr < _internalInputData.GetNR() - prevLayerErrors.GetNR() + 1; nr += _strideHeight) {
					long nco = 0;
					for (long nc = 0; nc < _internalInputData.GetNC() - prevLayerErrors.GetNC() + 1; nc += _strideWidth) {
						float_n error = 0;
						for (long i = 0; i < prevLayerErrors.GetNR(); i++) {
							for (long j = 0; j < prevLayerErrors.GetNC(); j++) {
								for (long prevLayerErrorsNumSamples = 0; prevLayerErrorsNumSamples < prevLayerErrors.GetNumSamples(); prevLayerErrorsNumSamples++) {
									error += _internalInputData[(((prevLayerErrorsNumSamples * _internalInputData.GetK()) + k) * _internalInputData.GetNR() + (nr + i)) * _internalInputData.GetNC() + (nc + j)] *
										prevLayerErrors[(((prevLayerErrorsNumSamples * prevLayerErrors.GetK()) + numOfFilters) * prevLayerErrors.GetNR() + i) * prevLayerErrors.GetNC() + j];
								}
							}
						}
						_dWeights[(((numOfFilters * _dWeights.GetK()) + k) * _dWeights.GetNR() + nro) * _dWeights.GetNC() + nco] = error;
						nco++;
					}
					nro++;
				}
			}
		}

		// Calculate gradient wrt. biases: (prevLayerErrors * 1)
		for (long numFilters = 0; numFilters < prevLayerErrors.GetK(); numFilters++) {
			float_n error = 0;
			for (long nr = 0; nr < prevLayerErrors.GetNR(); nr++) {
				for (long nc = 0; nc < prevLayerErrors.GetNC(); nc++) {
					for (long numSamples = 0; numSamples < prevLayerErrors.GetNumSamples(); numSamples++) {
						error += prevLayerErrors[(((numSamples * prevLayerErrors.GetK()) + numFilters) * prevLayerErrors.GetNR() + nr) * prevLayerErrors.GetNC() + nc];
					}
				}
			}
			_dBiases[numFilters] = error;
		}

		// Calculate gradient wrt. input: (prevLayerErrors * rotated180(_weights))
		_rotatedWeights.Flip180(_weights);

		long paddingWidth = _rotatedWeights.GetNC() - 1;
		long paddingHeight = _rotatedWeights.GetNR() - 1;
		Tensor convPrevLayerErrors;
		AddPadding(prevLayerErrors, convPrevLayerErrors, paddingWidth, paddingHeight);

		for (long numSamples = 0; numSamples < convPrevLayerErrors.GetNumSamples(); numSamples++) {
			for (long k = 0; k < _rotatedWeights.GetK(); k++) {
				long nro = 0;
				for (long nr = 0; nr < convPrevLayerErrors.GetNR() - _rotatedWeights.GetNR() + 1; nr += _strideHeight) {
					long nco = 0;
					for (long nc = 0; nc < convPrevLayerErrors.GetNC() - _rotatedWeights.GetNC() + 1; nc += _strideWidth) {
						float_n error = 0;
						for (long i = 0; i < _rotatedWeights.GetNR(); i++) {
							for (long j = 0; j < _rotatedWeights.GetNC(); j++) {
								for (long numOfFilters = 0; numOfFilters < convPrevLayerErrors.GetK(); numOfFilters++) {
									error += convPrevLayerErrors[(((numSamples * convPrevLayerErrors.GetK()) + numOfFilters) * convPrevLayerErrors.GetNR() + (nr + i)) * convPrevLayerErrors.GetNC() + (nc + j)]
										* _rotatedWeights[(((numOfFilters * _rotatedWeights.GetK()) + k) * _rotatedWeights.GetNR() + i) * _rotatedWeights.GetNC() + j];
								}
							}
						}
						_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nro) * _layerErrors.GetNC() + nco] = error;
						nco++;
					}
					nro++;
				}
			}
		}
	}

	void ConvolutionalLayer::Serialize(Serializer& serializer) {
		serializer.AddParentNode(_layerID);
		serializer.SerializeTensor(_weights, _layerID, "weights");
		serializer.SerializeTensor(_biases, _layerID, "biases");
	}

	void ConvolutionalLayer::Deserialize(Serializer& serializer) {
		serializer.DeserializeTensor(_weights, _layerID, "weights");
		serializer.DeserializeTensor(_biases, _layerID, "biases");
	}

	void ConvolutionalLayer::AddPadding(const nexural::Tensor& input, nexural::Tensor& output, long paddingWidth, long paddingHeight) {
		long newNR = input.GetNR() + 2 * paddingHeight;
		long newNC = input.GetNC() + 2 * paddingWidth;
		output.Resize(input.GetNumSamples(), input.GetK(), newNR, newNC);

		for (long numSamples = 0; numSamples < output.GetNumSamples(); numSamples++) {
			for (long k = 0; k < output.GetK(); k++) {
				// Top and bottom bording
				for (long nr = 0; nr < paddingHeight; nr++) {
					for (long nc = 0; nc < newNC; nc++) {
						output[(((numSamples * output.GetK()) + k) * output.GetNR() + nr) * output.GetNC() + nc] = 0;
						output[(((numSamples * output.GetK()) + k) * output.GetNR() + (nr + (output.GetNR() - paddingHeight))) * output.GetNC() + nc] = 0;
					}
				}

				// Left and right bording
				for (long nr = paddingHeight; nr < newNR - paddingHeight; nr++) {
					for (long nc = 0; nc < paddingWidth; nc++) {
						output[(((numSamples * output.GetK()) + k) * output.GetNR() + nr) * output.GetNC() + nc] = 0;
						output[(((numSamples * output.GetK()) + k) * output.GetNR() + nr) * output.GetNC() + nc + (output.GetNC() - paddingWidth)] = 0;
					}
				}

				// Copy the data from input to output
				long i = 0;
				for (long nr = paddingHeight; nr < newNR - paddingHeight; nr++) {
					long j = 0;
					for (long nc = paddingWidth; nc < newNC - paddingWidth; nc++) {
						output[(((numSamples * output.GetK()) + k) * output.GetNR() + nr) * output.GetNC() + nc] =
							input[(((numSamples * input.GetK()) + k) * input.GetNR() + i) * input.GetNC() + j];
						j++;
					}
					i++;
				}
			}
		}
	}
}
