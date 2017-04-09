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

#include "network_trainer.h"
#include "network.h"
#include <iostream>

namespace nexural {

	NetworkTrainer::NetworkTrainer() :
		_maxNumEpochs(10000),
		_maxEpochsWithoutProgress(6),
		_minLearningRateThreshold(0.00001),
		_batchSize(1),
		_solver(new SGDMomentum()),
		_beVerbose(true)
	{ };

	NetworkTrainer::NetworkTrainer(const std::string& trainerConfigPath) {
		InitTrainer(trainerConfigPath);
	}

	NetworkTrainer::~NetworkTrainer() {

	}

	void NetworkTrainer::InitTrainer(const std::string& trainerConfigPath) {
		Params trainerParams, solverParams;
		ConfigReader::DecodeTrainerCongif(trainerConfigPath, trainerParams, solverParams);

		_maxNumEpochs = parser::ParseInt(trainerParams, "max_num_epochs");
		_maxEpochsWithoutProgress = parser::ParseInt(trainerParams, "max_epochs_without_progress");
		_minLearningRateThreshold = parser::ParseFloat(trainerParams, "min_learning_rate_threshold");
		_minValidationErrorThreshold = parser::ParseFloat(trainerParams, "min_validation_error_threshold");
		_batchSize = parser::ParseInt(trainerParams, "batch_size");
		_beVerbose = parser::ParseBool(trainerParams, "be_verbose");

		// Init the solver
		std::string solverAlgorithm = parser::ParseString(solverParams, "algorithm");
		if (solverAlgorithm == "sgd") {
			_solver.reset(new SGD(solverParams));
		} else if (solverAlgorithm == "sgd_momentum") {
			_solver.reset(new SGDMomentum(solverParams));
		}
	}

	void NetworkTrainer::Train(Network& net, Tensor& trainingData, Tensor& targetData, const long batchSize) {
		Tensor *error, *weights, *dWeights, *biases, *dBiases;
		float_n prevEpochError = std::numeric_limits<float_n>::max();
		float_n currentEpochError, learningRateDecay = 0.00001, diffErrorThreshold = 0.01;
		bool doTraining = true;
		long currentEpoch = 0, stepsWithoutAnyProgress = 0;

		std::cout << std::endl << "The engine is initializing the network for the training process." << std::endl << std::endl;
		InitLayersForTraining(net);

		std::cout << "Starting the training process..." << std::endl;
		while (doTraining) {
			long trainingDataIter = trainingData.GetNumSamples();
			currentEpochError = 0;

			std::cout << "Current training epoch: " << currentEpoch << std::endl;
			
			for (int batchIndex = 0; batchIndex < trainingDataIter; batchIndex += batchSize) {
				_input.GetBatch(trainingData, batchIndex, batchSize);
				_target.GetBatch(targetData, batchIndex, batchSize);

				net._inputNetworkLayer->LoadData(_input);
				Tensor *internalNetData = net._inputNetworkLayer->GetOutput();

				// Feedforward the error
				for (auto it = net._computationalNetworkLyers.begin(); it < net._computationalNetworkLyers.end(); it++) {
 					(*it)->FeedForward(*internalNetData);
					internalNetData = (*it)->GetOutput();
				}

				net._lossNetworkLayer->FeedForward(*internalNetData);

				// Calculate the total error
				net._lossNetworkLayer->CalculateError(_target);
				net._lossNetworkLayer->CalculateTotalError(_target);
				error = net._lossNetworkLayer->GetLayerErrors();
				currentEpochError += net._lossNetworkLayer->GetTotalError();
				//std::cout << "Total error for current iteration: " << currentError << std::endl;

				// Backpropagate the error
				for (auto it = net._computationalNetworkLyers.rbegin(); it < net._computationalNetworkLyers.rend(); it++) {
					(*it)->BackPropagate(*error);
					error = (*it)->GetLayerErrors();
				}

				// Update the weights
				for (auto it = net._computationalNetworkLyers.rbegin(); it < net._computationalNetworkLyers.rend(); it++) {
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
			currentEpochError /= trainingDataIter;

			// If there isn't any progress, probably we are jumping over the global minimum
			// so, we need to reduce the learning rate in order to hit the minimum
			if (((prevEpochError - prevEpochError * diffErrorThreshold) < currentEpochError) && (currentEpochError < (prevEpochError + prevEpochError * diffErrorThreshold))) {
				stepsWithoutAnyProgress++;
				std::cout << " -!-[INFO] Num of steps without any progress: " << stepsWithoutAnyProgress << std::endl;
				if (stepsWithoutAnyProgress == _maxEpochsWithoutProgress) {
					std::cout << " -!-[INFO] Reducing the learning rate!" << std::endl;
					learningRateDecay += 0.00005;
					diffErrorThreshold *= 0.1;
					stepsWithoutAnyProgress = 0;
				}
			}
			else {
				stepsWithoutAnyProgress = 0;
			}

			// Print epoch error
			std::cout << " -- MEAN ERROR: " << currentEpochError << std::endl;

			//Update the learning rate
			std::cout << "   -- Learning rate before update: " << _solver->GetLearningRate() << std::endl;
			double learningRateStep = 1 / (1 + learningRateDecay * currentEpoch);
			_solver->UpdateLearningRate(learningRateStep);
			std::cout << "   -- Learning after before update: " << _solver->GetLearningRate() << std::endl;
			std::cout << std::endl << std::endl;

			prevEpochError = currentEpochError;
			currentEpoch++;

			if (currentEpoch == _maxNumEpochs) {
				doTraining = false;
				std::cout << std::endl << "[STOP CONDITION] The trainer has reached the maxim number of epochs!" << std::endl << std::endl;
			}
			else if (_solver->GetLearningRate() < _minLearningRateThreshold) {
				doTraining = false;
				std::cout << std::endl << "[STOP CONDITION] The trainer has reached the minim learning rate threshold!" << std::endl << std::endl;
			}
		}
	}

	void NetworkTrainer::InitLayersForTraining(Network& net) {
		for (int i = 0; i < net._computationalNetworkLyers.size(); i++) {
			net._computationalNetworkLyers[i]->SetupLayerForTraining();
		}
		net._lossNetworkLayer->SetupLayerForTraining();
	}
}
