// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LOSS_BASE_LAYER
#define _NEXURALNET_DNN_LOSS_BASE_LAYER

#include <map>
#include "memory"

#include "i_layer.h"
#include "i_loss_layer.h"
#include "tensor.h"
#include "data_types.h"

namespace nexural {
	class LossBaseLayer : public ILayer, ILossLayer {
	public:
		LossBaseLayer() { 
		
		}

		LossBaseLayer(const LayerParams& layerParams) { 
			_layerParams = layerParams;
		}

		virtual ~LossBaseLayer() { 
		
		}

		virtual void Setup(LayerShape& prevLayerShape) {

		}

		virtual void FeedForward(const Tensor& inputData) {
		
		}

		Tensor* GetOutput() {
			return &_outputData;
		}


		virtual Tensor* GetLayerErrors() {
			return &_layerErrors;
		}

		LayerShape GetOutputShape() {
			return _outputShape;
		}

		virtual void CalculateError() {

		}

	protected:
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
		Tensor _layerErrors;
		LayerParams _layerParams;

	};
	typedef std::shared_ptr<LossBaseLayer> LossBaseLayerPtr;
}
#endif 
