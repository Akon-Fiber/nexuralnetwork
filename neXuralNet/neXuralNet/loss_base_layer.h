// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LOSS_BASE_LAYER
#define _NEXURALNET_DNN_LOSS_BASE_LAYER

#include <map>
#include "memory"

#include "i_loss_layer.h"
#include "tensor.h"
#include "data_types.h"

namespace nexural {

	class LossBaseLayer : public ILossLayer {
	public:
		LossBaseLayer() { }
		LossBaseLayer(std::map<std::string, std::string> &layerParams) { }
		virtual ~LossBaseLayer() { }

		virtual void FeedForward(const Tensor& inputData) { }

		virtual Tensor& GetOutput() {
			return _outputData;
		}

		virtual LossResultBasePtr GetResultType() {
			return NULL;
		}

	protected:
		Tensor _outputData;
	};
	typedef std::shared_ptr<LossBaseLayer> LossBaseLayerPtr;
}
#endif 
