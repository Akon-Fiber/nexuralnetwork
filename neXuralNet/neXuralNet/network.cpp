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

#include "network.h"
#include <iostream>

namespace nexural {

	Network::Network(const std::string jsonFilePath) {
		NetworkReader netParser(jsonFilePath);
		netParser.loadNetwork(*this);

		LayerShape prevLayerShape = _inputNetworkLayer->GetOutputShape();
		for (int i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->Setup(prevLayerShape);
			prevLayerShape = _computationalNetworkLyers[i]->GetOutputShape();
		}
		_lossNetworkLayer->Setup(prevLayerShape);
	}

	Network::~Network() {

	}

	void Network::Run(Tensor& inputData) {
		_inputNetworkLayer->LoadData(inputData);
		Tensor *internalNetData = _inputNetworkLayer->GetOutput();

		for (int i = 0; i < _computationalNetworkLyers.size(); i++) {
			_computationalNetworkLyers[i]->FeedForward(*internalNetData);
			internalNetData = _computationalNetworkLyers[i]->GetOutput();
		}

		_lossNetworkLayer->FeedForward(*internalNetData);
		internalNetData = _lossNetworkLayer->GetOutput();

		// TODO: Delete this 
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


}

