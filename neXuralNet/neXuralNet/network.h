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

#include <vector>

#include "input_layers.h"
#include "computational_layers.h"
#include "loss_layers.h"
#include "utils.h"
#include "network_trainer.h"
#include "data_reader.h"
#include "config_reader.h"

#ifndef _NEXURALNET_DNN_NETWORK_NETWORK
#define _NEXURALNET_DNN_NETWORK_NETWORK

namespace nexural {
	class Network {
		friend class NetworkTrainer;
		typedef InputBaseLayerPtr InputNetworkLayer;
		typedef std::vector<ComputationalBaseLayerPtr> ComputationalNetworkLayers;
		typedef LossBaseLayerPtr LossNetworkLayer;

	public:
		Network() = delete;
		Network(const std::string networkConfigPath);
		explicit Network(const Network& network) = delete;
		explicit Network(Network&& network) = delete;
		Network& operator=(const Network& network) = delete;
		Network& operator=(Network&& network) = delete;
		~Network();

		void Run(Tensor& inputData);
		void Serialize(const std::string& dataPath);
		void Deserialize(const std::string& dataPath);

	private:
		void SetInputLayer(InputBaseLayerPtr inputLayer);
		void AddComputationalLayer(ComputationalBaseLayerPtr computationalLayer);
		void SetLossLayer(LossBaseLayerPtr lossLayer);
		void InitNetwork(const std::string networkConfigPath);

	private:
		InputNetworkLayer _inputNetworkLayer;
		ComputationalNetworkLayers _computationalNetworkLyers;
		LossNetworkLayer _lossNetworkLayer;
	};
}
#endif
