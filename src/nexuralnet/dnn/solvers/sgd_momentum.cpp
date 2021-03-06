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

#include "sgd_momentum.h"

namespace nexural {
	SGDMomentum::SGDMomentum() : BaseSolver(),
		_mu(0.9) { }

	SGDMomentum::SGDMomentum(Params &solverParams) : BaseSolver(solverParams) {
		_mu = parser::ParseFloat(solverParams, "momentum");
	}

	SGDMomentum::~SGDMomentum() {

	}

	void SGDMomentum::UpdateWeights(Tensor& weights, const Tensor& dWeights, const std::string& layerID) {
		Tensor v;
		auto search = _weightsVelocity.find(layerID);

		if (search != _weightsVelocity.end()) {
			v.ShareTensor(search->second);
		}
		else {
			v.Resize(weights.GetShape());
			v.Fill(0);
			_weightsVelocity.insert({ layerID, std::ref(v) });
		}

		// Momentum update
		// Formula: V = mu * V - learning_rate * (dW + W * weight_decay);
		for (int i = 0; i < weights.Size(); i++) {
			v[i] = _mu * v[i] + _learningRate * (dWeights[i] + weights[i] * _weightDecay);
			weights[i] -= v[i];
		}
	}

	void SGDMomentum::UpdateBiases(Tensor& baises, const Tensor& dBiases, const std::string& layerID) {
		Tensor v;
		auto search = _biasesVelocity.find(layerID);

		if (search != _biasesVelocity.end()) {
			v.ShareTensor(search->second);
		}
		else {
			v.Resize(baises.GetShape());
			v.Fill(0);
			_biasesVelocity.insert({ layerID, std::ref(v) });
		}

		for (int i = 0; i < baises.Size(); i++) {
			v[i] = _mu * v[i] + _learningRate * dBiases[i];
			baises[i] -= v[i];
		}
	}
}
