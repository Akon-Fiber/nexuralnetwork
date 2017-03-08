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

#ifndef _NEXURALNET_DNN_NETWORK_NETWORK
#define _NEXURALNET_DNN_NETWORK_NETWORK

namespace nexural {
	template <typename INPUT_LAYER_TYPE, typename INPUT_LAYER_DATA_TYPE>
	class Network {
		typedef std::vector<ComputationalBaseLayerPtr> ComputationalNetworkLayers;
		typedef LossBaseLayerPtr LossNetworkLayer;

	public:
		Network() = delete;

		Network(const std::string jsonFilePath) {
			NetworkReader netParser(jsonFilePath);
			netParser.loadNetwork(*this);

			LayerShape prevLayerShape = _inputNetworkLayer.GetOutputShape();
			for (int i = 0; i < this->_computationalNetworkLyers.size(); i++) {
				_computationalNetworkLyers[i]->Setup(prevLayerShape);
				prevLayerShape = _computationalNetworkLyers[i]->GetOutputShape();
			}
			_lossNetworkLayer->Setup(prevLayerShape);
		}

		~Network() {

		}

		void InitInputLayer(const LayerParams &layerParams) {
			_inputNetworkLayer.Init(layerParams);
		}

		void SetLossLayer(const std::string lossLayerType, LayerParams &layerParams) {
			if("mse") {
				_lossNetworkLayer.reset(new MSELossLayer(layerParams));
			}
		}

		void AddComputationalLayer(ComputationalBaseLayerPtr layer) {
			_computationalNetworkLyers.push_back(layer);
		}

		void Run(INPUT_LAYER_DATA_TYPE inputData) {
			_inputNetworkLayer.LoadData(inputData);
			Tensor *internalNetData = _inputNetworkLayer.GetOutput();

			// Computational layers
			for (int i = 0; i < this->_computationalNetworkLyers.size(); i++) {
				_computationalNetworkLyers[i]->FeedForward(*internalNetData);
				internalNetData = _computationalNetworkLyers[i]->GetOutput();	
			}
			
			// Loss layers
			_lossNetworkLayer->FeedForward(*internalNetData);
			internalNetData = _lossNetworkLayer->GetOutput();

			// TODO: Delete this 
			for (int i = 0; i < internalNetData->Size(); i++) {
				std::cout << (*(&(*internalNetData)))[i] << std::endl;
			}
		}

		void Train() {

		}

	private:

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
					throw std::runtime_error("InputSettings member is missing from the JSON file!");
				}

				const rapidjson::Value& networkLayers = _document["NetworkLayers"];

				if (!networkLayers.HasMember("InputLayerSettings")) {
					throw std::runtime_error("InputLayerSettings member is missing from the JSON file!");
				}

				if (!networkLayers.HasMember("ComputationalLayers")) {
					throw std::runtime_error("ComputationalLayers member is missing from the JSON file!");
				}

				if (!networkLayers.HasMember("LossLayerSettings")) {
					throw std::runtime_error("LossLayerSettings member is missing from the JSON file!");
				}

				if (!_document.HasMember("TrainerSettings")) {
					throw std::runtime_error("TrainerSettings member is missing from the JSON file!");
				}

				// Input layer setup
				const rapidjson::Value& inputSettings = networkLayers["InputLayerSettings"];
				LayerParams inputLayerParams;
				for (rapidjson::Value::ConstMemberIterator iter = inputSettings.MemberBegin(); iter != inputSettings.MemberEnd(); ++iter) {
					inputLayerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
				}
				net.InitInputLayer(inputLayerParams);

				// Computational layers setup
				const rapidjson::Value& netLayers = networkLayers["ComputationalLayers"];
				for (rapidjson::SizeType i = 0; i < netLayers.Size(); i++)
				{
					const rapidjson::Value& currentLayer = netLayers[i];

					if (!currentLayer.HasMember("type")) {
						throw std::runtime_error("type member is missing from the JSON file!");
					}
					if (!currentLayer.HasMember("params")) {
						throw std::runtime_error("params member is missing from the JSON file!");
					}

					LayerParams layerParams;
					std::string type_member = currentLayer["type"].GetString();
					const rapidjson::Value& params_member = currentLayer["params"];

					for (rapidjson::Value::ConstMemberIterator iter = params_member.MemberBegin(); iter != params_member.MemberEnd(); ++iter) {
						layerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
					}

					if (type_member == "max_pooling") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::MaxPoolingLayer(layerParams)));
					}
					else if (type_member == "average_pooling") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::AveragePoolingLayer(layerParams)));
					}
					else if (type_member == "relu") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::ReluLayer(layerParams)));
					}
					else if (type_member == "dropout") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::DropoutLayer(layerParams)));
					}
					else if (type_member == "fully_connected") {
						net.AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::FullyConnectedLayer(layerParams)));
					}
				}

				// Loss layer setup
				const rapidjson::Value& lossSettings = networkLayers["LossLayerSettings"];
				LayerParams lossLayerParams;
				std::string lossLayerType = lossSettings["type"].GetString();
				const rapidjson::Value& lossSettingsParams = lossSettings["params"];
				for (rapidjson::Value::ConstMemberIterator iter = lossSettingsParams.MemberBegin(); iter != lossSettingsParams.MemberEnd(); ++iter) {
					lossLayerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
				}
				net.SetLossLayer(lossLayerType, inputLayerParams);
			}

		private:
			FILE* _fp;
			rapidjson::Document _document;
		};

	private:
		ComputationalNetworkLayers _computationalNetworkLyers;
		LossNetworkLayer _lossNetworkLayer;
		INPUT_LAYER_TYPE _inputNetworkLayer;
	};
}
#endif
