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

#include "../solvers/solvers.h"

#ifndef _NEXURALNET_DNN_NETWORK_NETWORK_TRAINNER
#define _NEXURALNET_DNN_NETWORK_NETWORK_TRAINNER

namespace nexural {
	class Network;

	class NetworkTrainer {
		typedef BaseSolverPtr NetSolver;

	public:
		NetworkTrainer();
		NetworkTrainer(const std::string& trainerConfigPath);
		~NetworkTrainer();

		void Train(Network& net, Tensor& trainingData, Tensor& validationData, Tensor& targetData, const long batchSize = 1);

	private:
		void InitTrainer(const std::string& trainerConfigPath);
		void InitLayersForTraining(Network& net);

	private:
		long _maxNumEpochs; 
		long _maxEpochsWithoutProgress;
		float_n _minLearningRateThreshold;
		float_n _minValidationErrorThreshold;
		float_n _updateLRThreshold;
		float_n _learningRateDecay;
		long _batchSize;
		Tensor _inputData;
		Tensor _targetData;
		Tensor _validationData;
		Tensor _validationTargetData;
		NetSolver _solver;
		bool _beVerbose;
	};
}
#endif