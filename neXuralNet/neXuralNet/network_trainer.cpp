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
		_maxNumEpochs(10),
		_maxNumEpochsWithoutProgress(100),
		_minErrorThreshold(0.0001f),
		_batchSize(1),
		_solver(new SGD()),
		_beVerbose(true)
	{ };

	NetworkTrainer::NetworkTrainer(const std::string trainerConfigPath) {
		InitTrainer(trainerConfigPath);
	}

	NetworkTrainer::~NetworkTrainer() {

	}

	void NetworkTrainer::InitTrainer(const std::string trainerConfigPath) {
		TrainerSettings trainerSettings;
		ConfigReader::DecodeTrainerCongif(trainerConfigPath, trainerSettings);

		_maxNumEpochs = parser::ParseInt(trainerSettings, "max_num_epochs");
		_maxNumEpochsWithoutProgress = parser::ParseInt(trainerSettings, "max_num_epochs_without_progress");
		_minErrorThreshold = parser::ParseFloat(trainerSettings, "min_error_threshold");
		_batchSize = parser::ParseInt(trainerSettings, "batch_size");
		_beVerbose = parser::ParseBool(trainerSettings, "be_verbose");

		std::string selectedSolver = parser::ParseString(trainerSettings, "solver");
		// TODO: Check if memory is correctly deallocated
		if (selectedSolver == "sgd") {
			_solver.reset(new SGD());
		} else if (selectedSolver == "sgd_momentum") {
			_solver.reset(new SGDMomentum());
		}
	}

	void NetworkTrainer::Train(Network& net, Tensor& trainingData, Tensor& targetData, const long batchSize) {
		InitLayersForTraining(net);
		float currentError = 0;
		bool doTraining = true;
		long currentEpoch = 0;

		while (doTraining) {
			Tensor *error, *weights, *dWeights, *biases, *dBiases;
			//std::cout << "Current epoch: " << currentEpoch << std::endl << std::endl;
			long trainingDataIter = trainingData.GetNumSamples();
			for (int batchIndex = 0; batchIndex < trainingDataIter; batchIndex += batchSize) {
				//std::cout << "Iter: " << batchIndex << std::endl;
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
				currentError = net._lossNetworkLayer->GetTotalError();
				//std::cout << "Total error: " << currentError << std::endl;
				//std::cout << "-------------------------------" << currentError << std::endl << std::endl;

				if (currentError <= _minErrorThreshold) {
					//doTraining = false;
					//break;
				}

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
						_solver->UpdateWeights(*weights, *dWeights);
					}
					if ((*it)->HasBiases()) {
						biases = (*it)->GetLayerBiases();
						dBiases = (*it)->GetLayerDBiases();
						_solver->UpdateWeights(*biases, *dBiases);
					}
				}
			}
			currentEpoch++;
			if (currentEpoch == _maxNumEpochs) {
				doTraining = false;
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
