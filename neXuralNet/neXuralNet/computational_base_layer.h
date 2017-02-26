// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef MAVNET_DNN_LAYERS_BASE_LAYER
#define MAVNET_DNN_LAYERS_BASE_LAYER

#include "memory"

#include "i_computational_layer.h"
#include "data_types.h"
#include "tensor.h"

namespace nexural {

	class ComputationalBaseLayer : public IComputationalLayer {
	public:
		ComputationalBaseLayer() { }

		ComputationalBaseLayer(LayerParams &layerParams) {
			_layerParams = layerParams;
		}

		virtual ~ComputationalBaseLayer() { }

		virtual void FeedForward(const Tensor& inputData) {

		}

		virtual void BackPropagate(const Tensor& layerErrors) {

		}

		virtual void Update(float rate) {
		
		}

		virtual Tensor* GetOutput() {
			return &_outputData;
		}

		virtual Tensor* GetLayerErrors() {
			return &_layerErrors;
		}

	protected:
		Tensor _outputData;
		Tensor _layerErrors;
		LayerParams _layerParams;

	};
	typedef std::shared_ptr<ComputationalBaseLayer> ComputationalBaseLayerPtr;
}
#endif 
