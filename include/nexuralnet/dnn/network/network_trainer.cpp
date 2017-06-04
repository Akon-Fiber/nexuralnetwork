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

		_maxNumEpochs = parser::ParseLong(trainerParams, "max_num_epochs");
		_maxEpochsWithoutProgress = parser::ParseLong(trainerParams, "max_epochs_without_progress");
		_minLearningRateThreshold = parser::ParseFloat(trainerParams, "min_learning_rate_threshold");
		_minValidationErrorThreshold = parser::ParseFloat(trainerParams, "min_validation_error_threshold");
		_updateLRThreshold = parser::ParseFloat(trainerParams, "update_learning_rate_threshold");
		_learningRateDecay = parser::ParseFloat(trainerParams, "learning_rate_decay");
		_batchSize = parser::ParseLong(trainerParams, "batch_size");
		_beVerbose = parser::ParseBool(trainerParams, "be_verbose");

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

	void NetworkTrainer::Train(Tensor& data, Tensor& labels) {
		float_n prevEpochError = std::numeric_limits<float_n>::max(), currentEpochError, validationError;
		long currentEpoch = 0, stepsWithoutAnyProgress = 0;
		bool doTraining = true;

		std::cout << std::endl << "Splitting data in training and validation datasets." << std::endl << std::endl;
		helper::SplitData(data, _trainingData, _validationData);
		helper::SplitData(labels, _trainingTargetData, _validationTargetData);
		data.Reset();
		labels.Reset();

		long trainingDataIterations = _trainingData.GetNumSamples();
		long validationDataIterations = _validationData.GetNumSamples();

		std::cout << "Starting the training process..." << std::endl;
		while (doTraining) {
			currentEpochError = 0;
			
			std::cout << "Current training epoch: " << currentEpoch << std::endl;

			// Shuffle the training set
			std::vector<long> batchesIndexes;
			for (int i = 0; i < trainingDataIterations; i += _batchSize) {
				batchesIndexes.push_back(i);
			}
			std::random_shuffle(batchesIndexes.begin(), batchesIndexes.end());

			for (size_t batchIndex = 0; batchIndex < batchesIndexes.size(); batchIndex ++) {
				_subTrainingData.GetBatch(_trainingData, batchesIndexes[batchIndex], _batchSize);
				_subTrainingTargetData.GetBatch(_trainingTargetData, batchesIndexes[batchIndex], _batchSize);

				_net._inputNetworkLayer->LoadData(_subTrainingData);
				Tensor *internalNetData = _net._inputNetworkLayer->GetOutput();

				for (auto it = _net._computationalNetworkLyers.begin(); it < _net._computationalNetworkLyers.end(); it++) {
					(*it)->FeedForward(*internalNetData);
					internalNetData = (*it)->GetOutput();

					// =============== DEBUG =====================================
					for (int i = 0; i < internalNetData->Size(); i++) {
						if (std::isnan((*(&(*internalNetData)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is nan in feedforward | Iter: " + std::to_string(trainingDataIterations));
						}
						else if (std::isinf((*(&(*internalNetData)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is inf in feedforward | Iter: " + std::to_string(trainingDataIterations));
						}
					}
					// =============== DEBUG END ==================================
				}

				_net._lossNetworkLayer->FeedForward(*internalNetData);
				_net._lossNetworkLayer->CalculateError(_subTrainingTargetData);
				_net._lossNetworkLayer->CalculateTotalError(_subTrainingTargetData);
				error = _net._lossNetworkLayer->GetLayerErrors();
				currentEpochError += _net._lossNetworkLayer->GetTotalError();

				// =============== DEBUG =====================================
				for (int i = 0; i < error->Size(); i++) {
					if (std::isnan((*(&(*error)))[i])) {
						throw std::runtime_error("Loss layer is nan in backprop | Iter: " + std::to_string(trainingDataIterations));
					}
					else if (std::isinf((*(&(*error)))[i])) {
						throw std::runtime_error("Loss layer is inf in backprop | Iter: " + std::to_string(trainingDataIterations));
					}
				}
				// =============== DEBUG END ==================================

				for (auto it = _net._computationalNetworkLyers.rbegin(); it < _net._computationalNetworkLyers.rend(); it++) {
					(*it)->BackPropagate(*error);
					error = (*it)->GetLayerErrors();

					// =============== DEBUG =====================================
					for (int i = 0; i < error->Size(); i++) {
						if (std::isnan((*(&(*error)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is nan in backprop | Iter: " + std::to_string(trainingDataIterations));
						}
						else if (std::isinf((*(&(*error)))[i])) {
							throw std::runtime_error((*it)->GetLayerID() + " is inf in backprop | Iter: " + std::to_string(trainingDataIterations));
						}
					}
					// =============== DEBUG END ==================================
				}

				// Update the weights and biases using the solver (only if the layer has weights and/or biases)
				// TODO: For each layer add weight_decay_multiplier
				for (auto it = _net._computationalNetworkLyers.rbegin(); it < _net._computationalNetworkLyers.rend(); it++) {
					if ((*it)->HasWeights()) {
						weights = (*it)->GetLayerWeights();
						dWeights = (*it)->GetLayerDWeights();
						_solver->UpdateWeights(*weights, *dWeights, (*it)->GetLayerID());
					}
					if ((*it)->HasBiases()) {
						biases = (*it)->GetLayerBiases();
						dBiases = (*it)->GetLayerDBiases();
						_solver->UpdateWeights(*biases, *dBiases, (*it)->GetLayerID());
					}
				}
			}
			currentEpochError /= trainingDataIterations;
			prevEpochError = currentEpochError;
			std::cout << " -- EPOCH MEAN ERROR: " << currentEpochError << std::endl;

			// Training validation
			validationError = 0;
			for (int validationBatchIndex = 0; validationBatchIndex < validationDataIterations; validationBatchIndex += _batchSize) {
				_subValidationData.GetBatch(_validationData, validationBatchIndex, _batchSize);
				_subValidationTargetData.GetBatch(_validationTargetData, validationBatchIndex, _batchSize);

				_net._inputNetworkLayer->LoadData(_subValidationData);
				Tensor *internalNetData = _net._inputNetworkLayer->GetOutput();
				for (auto it = _net._computationalNetworkLyers.begin(); it < _net._computationalNetworkLyers.end(); it++) {
					(*it)->FeedForward(*internalNetData);
					internalNetData = (*it)->GetOutput();
				}
				_net._lossNetworkLayer->FeedForward(*internalNetData);
				_net._lossNetworkLayer->CalculateTotalError(_subValidationTargetData);
				validationError += _net._lossNetworkLayer->GetTotalError();
			}
			validationError /= validationDataIterations;
			std::cout << " -- VALIDATION MEAN ERROR: " << validationError << std::endl;

			// If there isn't any progress, probably we are jumping over the global minimum
			// so, we need to reduce the learning rate in order to hit the minimum
			if (((prevEpochError - prevEpochError * _updateLRThreshold) < currentEpochError) && (currentEpochError < (prevEpochError + prevEpochError * _updateLRThreshold))) {
				stepsWithoutAnyProgress++;
				std::cout << " -! [INFO] Num of steps without any progress: " << stepsWithoutAnyProgress << std::endl;
				if (stepsWithoutAnyProgress == _maxEpochsWithoutProgress) {
					std::cout << " -! [INFO] Reducing the learning rate!" << std::endl;
					_learningRateDecay += 0.0005;
					_updateLRThreshold *= 0.1;
					stepsWithoutAnyProgress = 0;
				}
			}
			else {
				stepsWithoutAnyProgress = 0;
			}

			// Update the learning rate: at each epoch decrease the learning rate
			std::cout << "    -- Learning rate before update: " << _solver->GetLearningRate() << std::endl;
			double learningRateStep = 1 / (1 + _learningRateDecay * currentEpoch);
			_solver->UpdateLearningRate(learningRateStep);
			std::cout << "    -- Learning after update: " << _solver->GetLearningRate() << std::endl;
			std::cout << std::endl << std::endl;

			currentEpoch++;

			if (currentEpoch == _maxNumEpochs) {
				doTraining = false;
				std::cout << std::endl << "[STOP CONDITION] The trainer has reached the maxim number of epochs!" << std::endl << std::endl;
			}
			else if (validationError < _minValidationErrorThreshold) {
				doTraining = false;
				std::cout << std::endl << "[STOP CONDITION] The trainer has reached the minim validation threshold!" << std::endl << std::endl;
			}
			else if (_solver->GetLearningRate() < _minLearningRateThreshold) {
				doTraining = false;
				std::cout << std::endl << "[STOP CONDITION] The trainer has reached the minim learning rate threshold!" << std::endl << std::endl;
			}
		}
	}

	void NetworkTrainer::Train(const std::string& dataFolderPath, const std::string& labelsFilePath, const TrainingDataSource trainingDataSource, const TargetDataSource targetDataSource, const bool saveNetworkInfo, const std::string& networkInfoFilePath) {
		Tensor data, labels;

		_saveNetworkInfo = saveNetworkInfo;
		_networkInfoFilePath = networkInfoFilePath;

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

		Train(data, labels);
	}

	void NetworkTrainer::Serialize(const std::string& dataPath) {
		_net.Serialize(dataPath);
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
}
