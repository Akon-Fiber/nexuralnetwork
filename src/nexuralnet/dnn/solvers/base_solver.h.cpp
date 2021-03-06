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

#include "base_solver.h"

namespace nexural {
	BaseSolver::BaseSolver() : _learningRate(0.01), _weightDecay(0.0005) { }

	BaseSolver::BaseSolver(Params &solverParams) {
		_learningRate = parser::ParseFloat(solverParams, "learning_rate");
		_weightDecay = parser::ParseFloat(solverParams, "weight_decay");
	}

	BaseSolver::~BaseSolver() {

	}

	void BaseSolver::UpdateLearningRate(const float_n scaleFactor) {
		_learningRate *= scaleFactor;
	}

	float_n BaseSolver::GetLearningRate() const {
		return _learningRate;
	}

}
