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

#include <iostream>
#include "network_trainer.h"
#include "network.h"
#include "../../tools/data_reader.h"

namespace nexural {
	NetworkTrainer::NetworkTrainer(const std::string& networkConfigSource, const std::string& trainerConfigSource, const ConfigSourceType& configSourceType) {
		InitTrainer(networkConfigSource, trainerConfigSource, configSourceType);
	}

	NetworkTrainer::~NetworkTrainer() {

	}

	void NetworkTrainer::InitTrainer(const std::string& networkConfigSource, const std::string& trainerConfigSource, const ConfigSourceType& configSourceType) {
		Params trainerParams, solverParams;
		ConfigReader::DecodeTrainerCongif(trainerConfigSource, trainerParams, solverParams, configSourceType);

		// Init parameters from config
		_maxNumEpochs = parser::ParseLong(trainerParams, "max_num_epochs");
		_minLearningRateThreshold = parser::ParseFloat(trainerParams, "min_learning_rate_threshold");
		_minValidationErrorThreshold = parser::ParseFloat(trainerParams, "min_validation_error_threshold");
		_learningRateDecay = parser::ParseFloat(trainerParams, "learning_rate_decay");
		_batchSize = parser::ParseLong(trainerParams, "batch_size");
		_trainingDatasetPercentage = parser::ParseFloat(trainerParams, "training_dataset_percentage");
		_autosaveTrainingNumEpochs = parser::ParseLong(trainerParams, "autosave_training_num_epochs");

		// Init the solver
		std::string solverAlgorithm = parser::ParseString(solverParams, "algorithm");
		if (solverAlgorithm == "sgd") {
			_solver.reset(new SGD(solverParams));
		} else if (solverAlgorithm == "sgd_momentum") {
			_solver.reset(new SGDMomentum(solverParams));
		}

		_net.CreateNetworkLayers(networkConfigSource, configSourceType);
		SetInputBatchSize(_batchSize);
		_net.SetupNetwork();
		InitLayersForTraining();
	}

	void NetworkTrainer::Train(Tensor& data, Tensor& labels, const std::string& outputTrainedDataFilePath, const std::string& outputTrainerInfoFolderPath) {
		_trainerInfoFilePath = outputTrainerInfoFolderPath + "trainerInfo.json";

		float_n prevEpochError = std::numeric_limits<float_n>::max();
		long currentEpoch = 1;
		bool doTraining = true;
		
		InitConfusionMatrices();

		// Splitting data in training and validation datasets
		helper::SplitData(data, _trainingData, _validationData, _trainingDatasetPercentage);
		helper::SplitData(labels, _trainingTargetData, _validationTargetData, _trainingDatasetPercentage);
		data.Reset();
		labels.Reset();

		long trainingDataIterations = _trainingData.GetNumSamples();
		long validationDataIterations = _validationData.GetNumSamples();

		_net.Serialize(outputTrainerInfoFolderPath + "initial_weights.json");
		_trainerInfoWriter.Write("result_type", helper::NetworkResultTypeToString(_net._lossNetworkLayer->GetResultType()));

		while (doTraining) {
			float_n currentEpochError = 0, validationError = 0;
			_trainerInfoWriter.AddEpoch(currentEpoch);

			// Shuffle the training dataset
			std::vector<long> batchesIndexes;
			for (int i = 0; i < trainingDataIterations; i += _batchSize) {
				batchesIndexes.push_back(i);
			}
			std::random_shuffle(batchesIndexes.begin(), batchesIndexes.end());

			// Split in batches (if needed) and start learning
			size_t iterationsDone = 0;
			for (size_t batchIndex = 0; batchIndex < batchesIndexes.size(); batchIndex++) {
				_subTrainingData.GetBatch(_trainingData, batchesIndexes[batchIndex], _batchSize);
				_subTrainingTargetData.GetBatch(_trainingTargetData, batchesIndexes[batchIndex], _batchSize);

				_net._inputNetworkLayer->LoadData(_subTrainingData);
				Tensor *internalNetData = _net._inputNetworkLayer->GetOutput();

				for (auto it = _net._computationalNetworkLyers.begin(); it < _net._computationalNetworkLyers.end(); it++) {
					(*it)->FeedForward(*internalNetData, NetworkState::TRAINING);
					internalNetData = (*it)->GetOutput();

#ifdef _DEBUG_NEXURAL_TRAINER
					for (int i = 0; i < internalNetData->Size(); i++) {
						if (std::isnan((*(&(*internalNetData)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is nan in feedforward | Iter: " + std::to_string(trainingDataIterations));
						}
						else if (std::isinf((*(&(*internalNetData)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is inf in feedforward | Iter: " + std::to_string(trainingDataIterations));
						}
					}
#endif
				}

				_net._lossNetworkLayer->FeedForward(*internalNetData, NetworkState::TRAINING);
				_net._lossNetworkLayer->CalculateError(_subTrainingTargetData);
				_error = _net._lossNetworkLayer->GetLayerErrors();
				_net._lossNetworkLayer->CalculateTrainingMetrics(_subTrainingTargetData, _trainingConfusionMatrix);
				currentEpochError += _net._lossNetworkLayer->GetTotalError();
				iterationsDone++;
				

#ifdef _DEBUG_NEXURAL_TRAINER
				for (int i = 0; i < error->Size(); i++) {
					if (std::isnan((*(&(*error)))[i])) {
						throw std::runtime_error("Loss layer is nan in backprop | Iter: " + std::to_string(trainingDataIterations));
					}
					else if (std::isinf((*(&(*error)))[i])) {
						throw std::runtime_error("Loss layer is inf in backprop | Iter: " + std::to_string(trainingDataIterations));
					}
				}
#endif

				for (auto it = _net._computationalNetworkLyers.rbegin(); it < _net._computationalNetworkLyers.rend(); it++) {
					(*it)->BackPropagate(*_error);
					_error = (*it)->GetLayerErrors();

#ifdef _DEBUG_NEXURAL_TRAINER
					for (int i = 0; i < error->Size(); i++) {
						if (std::isnan((*(&(*error)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is nan in backprop | Iter: " + std::to_string(trainingDataIterations));
						}
						else if (std::isinf((*(&(*error)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is inf in backprop | Iter: " + std::to_string(trainingDataIterations));
						}
					}
#endif
				}

				// Update the weights and biases using the solver (only if the layer has weights and/or biases)
				for (auto it = _net._computationalNetworkLyers.rbegin(); it < _net._computationalNetworkLyers.rend(); it++) {
					if ((*it)->HasWeights()) {
						_weights = (*it)->GetLayerWeights();
						_dWeights = (*it)->GetLayerDWeights();
						_solver->UpdateWeights(*_weights, *_dWeights, (*it)->GetLayerID());
					}
					if ((*it)->HasBiases()) {
						_biases = (*it)->GetLayerBiases();
						_dBiases = (*it)->GetLayerDBiases();
						_solver->UpdateWeights(*_biases, *_dBiases, (*it)->GetLayerID());
					}
				}
			}

			// Calculate mean error
			_currentEpochError = currentEpochError / iterationsDone;

			// Training validation
			iterationsDone = 0;
			for (int validationBatchIndex = 0; validationBatchIndex < validationDataIterations; validationBatchIndex += _batchSize) {
				_subValidationData.GetBatch(_validationData, validationBatchIndex, _batchSize);
				_subValidationTargetData.GetBatch(_validationTargetData, validationBatchIndex, _batchSize);

				_net._inputNetworkLayer->LoadData(_subValidationData);
				Tensor *internalNetData = _net._inputNetworkLayer->GetOutput();
				for (auto it = _net._computationalNetworkLyers.begin(); it < _net._computationalNetworkLyers.end(); it++) {
					(*it)->FeedForward(*internalNetData, NetworkState::VALIDATION);
					internalNetData = (*it)->GetOutput();
				}
				_net._lossNetworkLayer->FeedForward(*internalNetData, NetworkState::VALIDATION);
				_net._lossNetworkLayer->CalculateTrainingMetrics(_subValidationTargetData, _validationConfusionMatrix);
				validationError += _net._lossNetworkLayer->GetTotalError();
				iterationsDone++;
			}

			// Calculate validation mean error
			_validationError = validationError / iterationsDone;

			WriteEpochStats(currentEpoch);

			// Update the learning rate: at each epoch decrease the learning rate
			_trainerInfoWriter.WriteEpochDetails(currentEpoch, "learning_rate", std::to_string(_solver->GetLearningRate()));
			double learningRateStep = 1 / (1 + _learningRateDecay * currentEpoch);
			_solver->UpdateLearningRate(learningRateStep);

			// Check for a stop condition
			if (currentEpoch == _maxNumEpochs) {
				doTraining = false;
				_trainerInfoWriter.Write("stop_condition", "reached_max_epochs_number");
			}
			else if (_validationError < _minValidationErrorThreshold) {
				doTraining = false;
				_trainerInfoWriter.Write("stop_condition", "reached_min_validation_threshold");
			}
			else if (_solver->GetLearningRate() < _minLearningRateThreshold) {
				doTraining = false;
				_trainerInfoWriter.Write("stop_condition", "reached_min_learning_rate_threshold");
			}
			else 
			{
				// TODO: Add stop condition from commands file
			}

			// Write progress on disk and save epoch's weights
			_trainerInfoWriter.Save(_trainerInfoFilePath);
			_net.Serialize(outputTrainerInfoFolderPath + "weights-epoch_" + std::to_string(currentEpoch) + ".json");

			// Save training - checkpoint
			if (currentEpoch % _autosaveTrainingNumEpochs == 0) {
				_net.Serialize(outputTrainedDataFilePath);
			}

			// Update or reset internal parameters
			prevEpochError = _currentEpochError;
			currentEpoch++;
			ResetConfusionMatrices();
		}
	}

	void NetworkTrainer::Train(const std::string& dataFolderPath, const std::string& labelsFilePath, const std::string& outputTrainedDataFilePath, const std::string& outputTrainerInfoFolderPath, const TrainingDataSource trainingDataSource, const TargetDataSource targetDataSource) {
		Tensor data, labels;

		switch (trainingDataSource) {
		case TrainingDataSource::IMAGES_DIRECTORY:
			tools::DataReader::ReadImagesFromDirectory(dataFolderPath, data);
			break;
		case TrainingDataSource::TXT_DATA_FILE:
			tools::DataReader::ReadTensorFromFile(dataFolderPath, data);
			break;
		case TrainingDataSource::MNIST_DATA_FILE:
			tools::DataReader::ReadMNISTData(dataFolderPath, data);
			break;
		default:
			break;
		}

		switch (targetDataSource) {
		case TargetDataSource::TXT_DATA_FILE:
			tools::DataReader::ReadTensorFromFile(labelsFilePath, labels);
			break;
		case TargetDataSource::MNIST_DATA_FILE:
			tools::DataReader::ReadMNISTLabels(labelsFilePath, labels);
			break;
		default:
			break;
		}

		Train(data, labels, outputTrainedDataFilePath, outputTrainerInfoFolderPath);
	}

	void NetworkTrainer::Serialize(const std::string& trainedDataFilePath) {
		_net.Serialize(trainedDataFilePath);
	}

	void NetworkTrainer::Deserialize(const std::string& dataPath) {
		_net.Deserialize(dataPath);
	}

	void NetworkTrainer::InitLayersForTraining() {
		for (size_t i = 0; i < _net._computationalNetworkLyers.size(); i++) {
			_net._computationalNetworkLyers[i]->SetupLayerForTraining();
		}
		_net._lossNetworkLayer->SetupLayerForTraining();
	}

	void NetworkTrainer::SetInputBatchSize(const long batchSize) {
		_net._inputNetworkLayer->SetInputBatchSize(batchSize);
	}

	void NetworkTrainer::InitConfusionMatrices() {
		NetworkResultType netType = _net._lossNetworkLayer->GetResultType();
		if (netType == NetworkResultType::REGRESSION) {
			_trainingConfusionMatrix.Resize(1, 1, 2, 2);
			_validationConfusionMatrix.Resize(1, 1, 2, 2);
			_trainingConfusionMatrix.Fill(0);
			_validationConfusionMatrix.Fill(0);
		}
		else if (netType == NetworkResultType::MULTICLASS_CLASSIFICATION) {
			long confusionMatrixSize = _net._lossNetworkLayer->GetOutputShape().GetNC();
			_trainingConfusionMatrix.Resize(1, 1, confusionMatrixSize, confusionMatrixSize);
			_validationConfusionMatrix.Resize(1, 1, confusionMatrixSize, confusionMatrixSize);
			_trainingConfusionMatrix.Fill(0);
			_validationConfusionMatrix.Fill(0);
		}
		else if (netType == NetworkResultType::BINARY_CLASSIFICATION) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the BINARY_CLASSIFICATION result type is not implemented!");
		}
		else if (netType == NetworkResultType::DETECTION) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the DETECTION result type is not implemented!");
		}
		else if (netType == NetworkResultType::UNKNOWN) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the result type is unknown!");
		}
	}

	void NetworkTrainer::ResetConfusionMatrices() {
		_trainingConfusionMatrix.Fill(0);
		_validationConfusionMatrix.Fill(0);
	}

	void NetworkTrainer::WriteEpochStats(const long currentEpoch) {
		_trainerInfoWriter.WriteEpochDetails(currentEpoch, "training_mean_error", std::to_string(_currentEpochError));
		_trainerInfoWriter.WriteEpochDetails(currentEpoch, "validation_mean_error", std::to_string(_validationError));

		NetworkResultType netType = _net._lossNetworkLayer->GetResultType();
		if (netType == NetworkResultType::REGRESSION) {
			
		}
		else if (netType == NetworkResultType::MULTICLASS_CLASSIFICATION) {

			_trainerInfoWriter.WriteEpochConfusionMatrix(currentEpoch, "training_confusion_matrix", _trainingConfusionMatrix);
			_trainerInfoWriter.WriteEpochConfusionMatrix(currentEpoch, "validation_confusion_matrix", _validationConfusionMatrix);
		}
		else if (netType == NetworkResultType::BINARY_CLASSIFICATION) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the BINARY_CLASSIFICATION result type is not implemented!");
		}
		else if (netType == NetworkResultType::DETECTION) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the DETECTION result type is not implemented!");
		}
		else if (netType == NetworkResultType::UNKNOWN) {
			throw std::runtime_error("Network trainer error: Can't init the confusion matrix because the result type is unknown!");
		}
	}
}
