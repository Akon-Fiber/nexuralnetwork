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
#include "network.h"

namespace nexural {
	Network::Network() { }

	Network::Network(const std::string networkConfigPath) {
		CreateNetworkLayers(networkConfigPath);
		SetupNetwork();
	}

	Network::~Network() {

	}

	void Network::Run(Tensor& inputData) {
		_inputNetworkLayer->LoadData(inputData);
		Tensor *internalNetData = _inputNetworkLayer->GetOutput();

		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->FeedForward(*internalNetData);
			internalNetData = _computationalNetworkLyers[i]->GetOutput();
		}

		_lossNetworkLayer->FeedForward(*internalNetData);
		internalNetData = _lossNetworkLayer->GetOutput();

		// TODO: Delete this 
		std::cout << "Result: " << std::endl;
		for (int i = 0; i < internalNetData->Size(); i++) {
			std::cout << (*(&(*internalNetData)))[i] << std::endl;
		}
	}

	void Network::SetInputLayer(InputBaseLayerPtr inputLayer) {
		_inputNetworkLayer = inputLayer;
	}

	void Network::AddComputationalLayer(ComputationalBaseLayerPtr computationalLayer) {
		_computationalNetworkLyers.push_back(computationalLayer);
	}

	void Network::SetLossLayer(LossBaseLayerPtr lossLayer) {
		_lossNetworkLayer = lossLayer;
	}

	void Network::CreateNetworkLayers(const std::string networkConfigPath) {
		LayerSettingsCollection layerSettingsCollection;
		ConfigReader::DecodeNetCongif(networkConfigPath, layerSettingsCollection);

		for (size_t i = 0; i < layerSettingsCollection.size(); i++) {
			std::string type_member = layerSettingsCollection[i].layerType;
			Params layerParams = layerSettingsCollection[i].layerParams;

			if (type_member == "bgr_image_input") {
				SetInputLayer(InputBaseLayerPtr(new nexural::BGRImageInputLayer(layerParams)));
			}
			else if (type_member == "gray_image_input") {
				SetInputLayer(InputBaseLayerPtr(new nexural::GrayImageInputLayer(layerParams)));
			}
			else if (type_member == "tensor_input") {
				SetInputLayer(InputBaseLayerPtr(new nexural::TensorInputLayer(layerParams)));
			}
			else if (type_member == "average_pooling") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::AveragePoolingLayer(layerParams)));
			}
			else if (type_member == "dynamic_average_pooling") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::DynamicAveragePoolingLayer(layerParams)));
			}
			else if (type_member == "max_pooling") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::MaxPoolingLayer(layerParams)));
			}
			else if (type_member == "dynamic_max_pooling") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::DynamicMaxPoolingLayer(layerParams)));
			}
			else if (type_member == "relu") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::ReluLayer(layerParams)));
			}
			else if (type_member == "leaky_relu") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::LeakyReluLayer(layerParams)));
			}
			else if (type_member == "tanh") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::TanHLayer(layerParams)));
			}
			else if (type_member == "dropout") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::DropoutLayer(layerParams)));
			}
			else if (type_member == "convolutional") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::ConvolutionalLayer(layerParams)));
			}
			else if (type_member == "fully_connected") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::FullyConnectedLayer(layerParams)));
			}
			else if (type_member == "mse") {
				SetLossLayer(LossBaseLayerPtr(new nexural::MSELossLayer(layerParams)));
			}
			else if (type_member == "softmax") {
				SetLossLayer(LossBaseLayerPtr(new nexural::SoftmaxLossLayer(layerParams)));
			}
		}
	}

	void Network::SetupNetwork() {
		LayerShape prevLayerShape = _inputNetworkLayer->GetOutputShape();
		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->Setup(prevLayerShape, i);
			prevLayerShape = _computationalNetworkLyers[i]->GetOutputShape();
		}
		_lossNetworkLayer->Setup(prevLayerShape);
	}

	void Network::Serialize(const std::string& dataPath) {
		Serializer serializer(SerializerType::JSON);
		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->Serialize(serializer);
		}
		serializer.Save(dataPath);
	}
	
	void Network::Deserialize(const std::string& dataPath) {
		Serializer serializer(SerializerType::JSON, dataPath);
		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->Deserialize(serializer);
		}
	}
}

