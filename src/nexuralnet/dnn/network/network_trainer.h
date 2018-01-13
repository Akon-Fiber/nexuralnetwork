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

#include "../solvers/solvers.h"
#include "../utility/trainer_info_writer/trainer_info_writer.h"

#ifndef NEXURALNET_DNN_NETWORK_NETWORK_TRAINNER
#define NEXURALNET_DNN_NETWORK_NETWORK_TRAINNER

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
		void Train(Tensor& data, Tensor& labels, const std::string& outputTrainedDataFilePath, const std::string& outputTrainerInfoFolderPath);
		void Train(const std::string& dataFolderPath, const std::string& labelsFilePath, const std::string& outputTrainedDataFilePath, const std::string& outputTrainerInfoFolderPath, const TrainingDataSource trainingDataSource, const TargetDataSource targetDataSource);
		void Serialize(const std::string& trainedDataFilePath);
		void Deserialize(const std::string& dataPath);

	private:
		void InitTrainer(const std::string& networkConfigPath, const std::string& trainerConfigPath, const ConfigSourceType& configSourceType);
		void InitLayersForTraining();
		void SetInputBatchSize(const long batchSize);
		void InitConfusionMatrices();
		void ResetConfusionMatrices();
		void WriteEpochStats(const long currentEpoch);

	private:
		Network _net;
		long _maxNumEpochs; 
		float_n _minLearningRateThreshold;
		float_n _minValidationErrorThreshold;
		float_n _learningRateDecay;
		long _batchSize;
		float_n _trainingDatasetPercentage;
		long _autosaveTrainingNumEpochs;
		Tensor _trainingData, _trainingTargetData, _subTrainingData, _subTrainingTargetData;
		Tensor _validationData, _validationTargetData, _subValidationData, _subValidationTargetData;
		Tensor *_error, *_weights, *_dWeights, *_biases, *_dBiases;
		NetSolver _solver;

		// Trainer info
		TrainerInfoWriter _trainerInfoWriter;
		std::string _trainerInfoFilePath;

		// Params for stats
		Tensor _trainingConfusionMatrix;
		Tensor _validationConfusionMatrix;
		float_n _currentEpochError;
		float_n _validationError;
		size_t _numOfIterations;
	};
}
#endif