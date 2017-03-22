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
#include <memory>
#include <string>
#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"

#include "input_layers.h"
#include "computational_layers.h"
#include "loss_layers.h"
#include "data_to_tensor_converter.h"
#include "network_trainer.h"
#include "data_reader.h"

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
		explicit Network(const Network& network) = delete;
		explicit Network(Network&& network) = delete;
		Network& operator=(const Network& network) = delete;
		Network& operator=(Network&& network) = delete;

		Network(const std::string jsonFilePath);
		~Network();

		void Run(Tensor& inputData);

	private:
		void SetInputLayer(InputBaseLayerPtr inputLayer);

		void AddComputationalLayer(ComputationalBaseLayerPtr computationalLayer);

		void SetLossLayer(LossBaseLayerPtr lossLayer);

		class NetworkReader {
		public:
			NetworkReader(const std::string jsonFilePath) {
				_fp = fopen(jsonFilePath.c_str(), "rb");
				if (!_fp) {
					throw std::runtime_error("Cannot load the JSON!");
				}
				char readBuffer[65536];
				rapidjson::FileReadStream is(_fp, readBuffer, sizeof(readBuffer));
				_document.ParseStream(is);
			}

			~NetworkReader() {
				if (_fp) {
					fclose(_fp);
				}
			}

			void loadNetwork(Network &net) {
				if (!_document.HasMember("NetworkLayers")) {
					throw std::runtime_error("NetworkLayers member is missing from the JSON file!");
				}

				if (!_document.HasMember("TrainerSettings")) {
					throw std::runtime_error("TrainerSettings member is missing from the JSON file!");
				}

				const rapidjson::Value& networkLayers = _document["NetworkLayers"];

				for (rapidjson::SizeType i = 0; i < networkLayers.Size(); i++)
				{
					const rapidjson::Value& currentLayer = networkLayers[i];

					if (!currentLayer.HasMember("type")) {
						throw std::runtime_error("type member is missing from the JSON file!");
					}
					if (!currentLayer.HasMember("params")) {
						throw std::runtime_error("params member is missing from the JSON file!");
					}

					LayerParams layerParams;
					std::string type_member = currentLayer["type"].GetString();
					const rapidjson::Value& paramsMember = currentLayer["params"];

					for (rapidjson::Value::ConstMemberIterator iter = paramsMember.MemberBegin(); iter != paramsMember.MemberEnd(); ++iter) {
						layerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
					}

					if (type_member == "bgr_image_input") {
						net.SetInputLayer(InputBaseLayerPtr(new nexural::BGRImageInputLayer(layerParams)));
					}
					else if (type_member == "gray_image_input") {
						net.SetInputLayer(InputBaseLayerPtr(new nexural::GrayImageInputLayer(layerParams)));
					}
					else if (type_member == "tensor_input") {
						net.SetInputLayer(InputBaseLayerPtr(new nexural::TensorInputLayer(layerParams)));
					}
					else if (type_member == "max_pooling") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::MaxPoolingLayer(layerParams)));
					}
					else if (type_member == "average_pooling") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::AveragePoolingLayer(layerParams)));
					}
					else if (type_member == "relu") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::ReluLayer(layerParams)));
					}
					else if (type_member == "leaky_relu") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::LeakyReluLayer(layerParams)));
					}
					else if (type_member == "tanh") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::TanHLayer(layerParams)));
					}
					else if (type_member == "dropout") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::DropoutLayer(layerParams)));
					}
					else if (type_member == "fully_connected") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::FullyConnectedLayer(layerParams)));
					}
					else if (type_member == "mse") {
						net.SetLossLayer(LossBaseLayerPtr(new nexural::MSELossLayer(layerParams)));
					}
					else if (type_member == "rmse") {
						net.SetLossLayer(LossBaseLayerPtr(new nexural::RMSELossLayer(layerParams)));
					}
				}
			}

		private:
			FILE* _fp;
			rapidjson::Document _document;
		};

	private:
		InputNetworkLayer _inputNetworkLayer;
		ComputationalNetworkLayers _computationalNetworkLyers;
		LossNetworkLayer _lossNetworkLayer;
	};
}
#endif
