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

namespace nexural {

	NetworkTrainer::~NetworkTrainer() {

	}

	void NetworkTrainer::Train(Network& net, Tensor& trainingData, Tensor& targetData, const long batchSize) {
		InitLayersForTraining(net);
		Tensor input, target;
		Tensor *error, *weights, *dWeights;

		for (int i = 0; i < trainingData.GetNumSamples(); i += batchSize) {
			input.GetBatch(trainingData, i, batchSize);
			target.GetBatch(targetData, i, batchSize);

			net._inputNetworkLayer->LoadData(input);
			Tensor *internalNetData = net._inputNetworkLayer->GetOutput();

			// Feedforward the error
			for (int i = 0; i < net._computationalNetworkLyers.size(); i++) {
				net._computationalNetworkLyers[i]->FeedForward(*internalNetData);
				internalNetData = net._computationalNetworkLyers[i]->GetOutput();
			}
			net._lossNetworkLayer->FeedForward(*internalNetData);

			// Calculate the total error
			net._lossNetworkLayer->CalculateError(targetData);
			error = net._lossNetworkLayer->GetLayerErrors();

			// Backpropagate the error
			for (size_t i = net._computationalNetworkLyers.size(); i > 0; i--) {
				net._computationalNetworkLyers[i]->BackPropagate(*error);
				error = net._computationalNetworkLyers[i]->GetLayerErrors();
			}

			// Update the weights
			for (size_t i = net._computationalNetworkLyers.size(); i > 0; i--) {
				weights = net._computationalNetworkLyers[i]->GetLayerWeights();
				dWeights = net._computationalNetworkLyers[i]->GetLayerDWeights();
				_solver->Update(*weights, *dWeights);
			}
		}
	}

	void NetworkTrainer::InitLayersForTraining(Network& net) {
		for (int i = 0; i < net._computationalNetworkLyers.size(); i++) {
			net._computationalNetworkLyers[i]->SetupLayerForTraining();
		}
		net._lossNetworkLayer->SetupLayerForTraining();
	}

	void NetworkTrainer::ResetLayersGradients(Network& net) {

	}

}
