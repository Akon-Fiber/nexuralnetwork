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
	class Momentum : public BaseSolver {
	public:
		Momentum() : BaseSolver(), 
			_mu(0.9f) { }

		~Momentum() {

		}

		virtual void Update(Tensor& weights, const Tensor& dWeights) {
			// float_t V = mu * V - learning_rate * (dW[i] + W[i] * weight_decay);
			// W[i] += V;

			// Classical momentum :
			// vW(t + 1) = momentum*Vw(t) - learning_rate*gradient_F(W(t))
			//	W(t + 1) = W(t) + vW(t + 1)

			/*float v;
			for (int i = 0; i < weights.Size(); i++) {
				v = _mu * v - _learningRate * dWeights[i];
				weights[i] = weights[i] + v;
			}*/
		}

	private:
		float _mu;
	};
}
#endif
