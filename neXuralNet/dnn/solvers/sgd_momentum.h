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

#include "base_solver.h"

#ifndef _NEXURALNET_DNN_SOLVERS_MOMENTUM
#define _NEXURALNET_DNN_SOLVERS_MOMENTUM

namespace nexural {
	class SGDMomentum : public BaseSolver {
	public:
		SGDMomentum() : BaseSolver(),
			_mu(0.9) { }

		SGDMomentum(float_n learningRate, float_n weightDecay) : BaseSolver(learningRate, weightDecay), _mu(0.9) { }

		SGDMomentum(float_n learningRate, float_n weightDecay, float_n momentum) : BaseSolver(learningRate, weightDecay), _mu(momentum) { }

		~SGDMomentum() {

		}

		virtual void UpdateWeights(Tensor& weights, const Tensor& dWeights, const std::string& layerID) {
			Tensor v;
			auto search = _weightsVelocity.find(layerID);

			if (search != _weightsVelocity.end()) {
				v.ShareTensor(search->second);
			}
			else {
				v.Resize(weights.GetShape());
				v.Fill(0);
				_weightsVelocity.insert(std::pair<std::string, Tensor>(layerID, std::ref(v)));
			}
			
			// Momentum update
			// Formula: V = mu * V - learning_rate * (dW + W * weight_decay);
			for (int i = 0; i < weights.Size(); i++) {
				v[i] = _mu * v[i] + _learningRate * (dWeights[i] + weights[i] * _weightDecay);
				weights[i] -= v[i];
			}
		}

		virtual void UpdateBiases(Tensor& baises, const Tensor& dBiases, const std::string& layerID) {
			Tensor v;
			auto search = _biasesVelocity.find(layerID);

			if (search != _biasesVelocity.end()) {
				v.ShareTensor(search->second);
			}
			else {
				v.Resize(baises.GetShape());
				v.Fill(0);
				_biasesVelocity.insert(std::pair<std::string, Tensor>(layerID, std::ref(v)));
			}

			for (int i = 0; i < baises.Size(); i++) {
				v[i] = _mu * v[i] + _learningRate * dBiases[i];
				baises[i] -= v[i];
			}
		}

	private:
		float_n _mu;
		std::map<std::string, Tensor> _weightsVelocity;
		std::map<std::string, Tensor> _biasesVelocity;
	};
}
#endif
