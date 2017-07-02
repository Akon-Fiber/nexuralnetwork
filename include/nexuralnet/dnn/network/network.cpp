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
#include <opencv2\core\mat.hpp>
#include "network.h"
#include "../../tools/data_writer.h"
#include "../../tools/converter.h"

namespace nexural {
	Network::Network() { }

	Network::Network(const std::string& networkConfigSource, const ConfigSourceType& configSourceType) {
		CreateNetworkLayers(networkConfigSource, configSourceType);
		SetupNetwork();
	}

	Network::~Network() {

	}

	void Network::Run(Tensor& inputData) {
		_inputNetworkLayer->LoadData(inputData);
		Tensor *internalNetData = _inputNetworkLayer->GetOutput();

		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->FeedForward(*internalNetData, FeedForwardType::RUN);
			internalNetData = _computationalNetworkLyers[i]->GetOutput();
		}

		_lossNetworkLayer->FeedForward(*internalNetData, FeedForwardType::RUN);
		_lossNetworkLayer->SetResult();
	}

	void Network::Run(cv::Mat& inputImage) {
		Tensor auxTensor;
		converter::CvtMatToTensor(inputImage, auxTensor);
		Run(auxTensor);
	}

	DNNBaseResult* Network::GetResult() {
		return _lossNetworkLayer->GetResult();
	}

	const std::string& Network::GetResultJSON() {
		return _lossNetworkLayer->GetResultJSON();
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

	void Network::CreateNetworkLayers(const std::string& networkConfigSource, const ConfigSourceType& configSourceType) {
		LayerSettingsCollection layerSettingsCollection;
		ConfigReader::DecodeNetCongif(networkConfigSource, layerSettingsCollection, configSourceType);

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
			else if (type_member == "max_pooling") {
				AddComputationalLayer(ComputationalBaseLayerPtr(new nexural::MaxPoolingLayer(layerParams)));
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

	void Network::SaveFiltersImages(const std::string& outputFolderPath) {
		Tensor* currentTensor;
		std::string convLayer = "convolutional_layer";
		for (size_t i = 0; i < _computationalNetworkLyers.size(); i++) {
			std::string layerID = _computationalNetworkLyers[i]->GetLayerID();
			if (layerID.find(convLayer) != std::string::npos) {
				currentTensor = _computationalNetworkLyers[i]->GetOutput();
				tools::DataWriter::WriteTensorImages(outputFolderPath, *currentTensor, layerID, true);
			}
		}
	}
}
