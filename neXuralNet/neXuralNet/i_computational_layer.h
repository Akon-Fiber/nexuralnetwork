// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_I_COMPUTATIONAL_LAYER
#define _NEXURALNET_DNN_I_COMPUTATIONAL_LAYER

#include "tensor.h"

namespace nexural {

	class IComputationalLayer {
	public:
		virtual ~IComputationalLayer() { }
		virtual void Setup(LayerShape& prevLayerShape) = 0;
		virtual void FeedForward(const Tensor& inputData) = 0;
		virtual void BackPropagate(const Tensor& layerErrors) = 0;
		virtual void Update() = 0;
		virtual Tensor* GetOutput() = 0;
		virtual Tensor* GetLayerErrors() = 0;
	};

}
#endif
