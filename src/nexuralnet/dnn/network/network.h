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

#include <vector>
#include <algorithm>

#include "../layers/input_layers/input_layers.h"
#include "../layers/computational_layers/computational_layers.h"
#include "../layers/loss_layers/loss_layers.h"
#include "../utility/config_reader/config_reader.h"
#include "network_trainer.h"

#ifndef NEXURALNET_DNN_NETWORK_NETWORK
#define NEXURALNET_DNN_NETWORK_NETWORK

namespace cv {
	class Mat;
}

namespace nexural {
	class Network {
		friend class NetworkTrainer;
		typedef InputBaseLayerPtr InputNetworkLayer;
		typedef std::vector<ComputationalBaseLayerPtr> ComputationalNetworkLayers;
		typedef LossBaseLayerPtr LossNetworkLayer;

	public:
		Network(const std::string& networkConfigSource, const ConfigSourceType& configSourceType = ConfigSourceType::FROM_FILE);
		explicit Network(const Network& network) = delete;
		explicit Network(Network&& network) = delete;
		Network& operator=(const Network& network) = delete;
		Network& operator=(Network&& network) = delete;
		~Network();

		void Run(Tensor& inputData);
		void Run(cv::Mat& inputImage);
		DNNBaseResult* GetResult();
		const std::string& GetResultJSON();
		void Deserialize(const std::string& dataPath);
		void SaveFiltersImages(const std::string& outputFolderPath);

	private:
		Network();
		void Serialize(const std::string& dataPath);
		void CreateNetworkLayers(const std::string& networkConfigSource, const ConfigSourceType& configSourceType);
		void SetupNetwork();
		void SetInputLayer(InputBaseLayerPtr inputLayer);
		void AddComputationalLayer(ComputationalBaseLayerPtr computationalLayer);
		void SetLossLayer(LossBaseLayerPtr lossLayer);
		
	private:
		InputNetworkLayer _inputNetworkLayer;
		ComputationalNetworkLayers _computationalNetworkLyers;
		LossNetworkLayer _lossNetworkLayer;
	};
}
#endif
