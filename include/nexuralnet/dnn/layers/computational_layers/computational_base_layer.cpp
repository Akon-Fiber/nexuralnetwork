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
#include "../../utility/params_parser.h"

namespace nexural {
	ComputationalBaseLayer::ComputationalBaseLayer() {

	}

	ComputationalBaseLayer::ComputationalBaseLayer(const Params &layerParams) {
		_layerParams = layerParams;
		_hasWeights = false;
		_hasBiases = false;
	}

	ComputationalBaseLayer::~ComputationalBaseLayer() {

	}

	Tensor* ComputationalBaseLayer::GetOutput() {
		return &_outputData;
	}

	Tensor* ComputationalBaseLayer::GetLayerErrors() {
		return &_layerErrors;
	}

	LayerShape ComputationalBaseLayer::GetOutputShape() {
		return _outputShape;
	}

	Tensor* ComputationalBaseLayer::GetLayerWeights() {
		return &_weights;
	}

	Tensor* ComputationalBaseLayer::GetLayerDWeights() {
		return &_dWeights;
	}

	Tensor* ComputationalBaseLayer::GetLayerBiases() {
		return &_biases;
	}

	Tensor* ComputationalBaseLayer::GetLayerDBiases() {
		return &_dBiases;
	}

	bool ComputationalBaseLayer::HasWeights() {
		return _hasWeights;
	}

	bool ComputationalBaseLayer::HasBiases() {
		return _hasBiases;
	}

	std::string ComputationalBaseLayer::GetLayerID() const {
		return _layerID;
	}

	void ComputationalBaseLayer::Serialize(Serializer& serializer) {

	}

	void ComputationalBaseLayer::Deserialize(Serializer& serializer) {

	}

	void ComputationalBaseLayer::SetWeights(const std::vector<float_n>& values) {
		_weights.Fill(values);
	}

	void ComputationalBaseLayer::SetBiases(const std::vector<float_n>& values) {
		_biases.Fill(values);
	}
}
