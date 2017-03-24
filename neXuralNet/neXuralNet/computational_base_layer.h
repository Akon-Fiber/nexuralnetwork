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

#include "i_layer.h"
#include "i_computational_layer.h"
#include "data_serializer.h"

#ifndef _NEXURALNET_DNN_LAYERS_COMPUTATIONAL_BASE_LAYER
#define _NEXURALNET_DNN_LAYERS_COMPUTATIONAL_BASE_LAYER

namespace nexural {
	class ComputationalBaseLayer : public ILayer, public IComputationalLayer {
	public:
		ComputationalBaseLayer() { 
		
		}

		ComputationalBaseLayer(const LayerParams &layerParams) {
			_layerParams = layerParams;
			_hasWeights = false;
			_hasBiases = false;
		}

		virtual ~ComputationalBaseLayer() { 
		
		}

		virtual Tensor* GetOutput() {
			return &_outputData;
		}

		virtual Tensor* GetLayerErrors() {
			return &_layerErrors;
		}

		virtual LayerShape GetOutputShape() {
			return _outputShape;
		}

		virtual Tensor* GetLayerWeights() {
			return &_weights;
		}

		virtual Tensor* GetLayerDWeights() {
			return &_dWeights;
		}

		virtual Tensor* GetLayerBiases() {
			return &_biases;
		}

		virtual Tensor* GetLayerDBiases() {
			return &_dBiases;
		}

		virtual bool HasWeights() {
			return _hasWeights;
		}

		virtual bool HasBiases() {
			return _hasBiases;
		}

		virtual std::string GetLayerID() const {
			return _layerID;
		}

	protected:
		LayerParams _layerParams;
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
		Tensor _layerErrors;
		Tensor _weights;
		Tensor _dWeights;
		Tensor _biases;
		Tensor _dBiases;
		bool _hasWeights;
		bool _hasBiases;
		std::string _layerID;
	};
	typedef std::shared_ptr<ComputationalBaseLayer> ComputationalBaseLayerPtr;
}
#endif 
