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

// Protect the network include
#include "network.h"

namespace nexural {
	class NetworkTrainer {
		typedef BaseSolverPtr NetSolver;

	public:
		enum class TrainingDataSource {
			IMAGES_DIRECTORY = 0,
			TXT_DATA_FILE = 1,
			MNIST_DATA_FILE = 2
		};

		enum class TargetDataSource {
			TXT_DATA_FILE = 0,
			MNIST_DATA_FILE = 1
		};

	public:
		NetworkTrainer(const std::string& networkConfigSource, const std::string& trainerConfigSource, const ConfigSourceType& configSourceType = ConfigSourceType::FROM_FILE);
		~NetworkTrainer();
		void Train(Tensor& data, Tensor& labels);
		void Train(const std::string& dataFolderPath, const std::string& labelsFilePath, const TrainingDataSource trainingDataSource, const TargetDataSource targetDataSource, const bool saveNetworkInfo, const std::string& networkInfoFilePath);
			
		void Serialize(const std::string& dataPath);

	private:
		void InitTrainer(const std::string& networkConfigPath, const std::string& trainerConfigPath, const ConfigSourceType& configSourceType);
		void InitLayersForTraining();
		void SetInputBatchSize(const long batchSize);

	private:
		Network _net;
		long _maxNumEpochs; 
		long _maxEpochsWithoutProgress;
		float_n _minLearningRateThreshold;
		float_n _minValidationErrorThreshold;
		float_n _updateLRThreshold;
		float_n _learningRateDecay;
		long _batchSize;
		Tensor _trainingData, _trainingTargetData, _subTrainingData, _subTrainingTargetData;
		Tensor _validationData, _validationTargetData, _subValidationData, _subValidationTargetData;
		Tensor *error, *weights, *dWeights, *biases, *dBiases;
		NetSolver _solver;
		bool _beVerbose;
		std::string _networkInfoFilePath;
		bool _saveNetworkInfo;
	};
}
#endif