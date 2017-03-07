// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_I_LAYER
#define _NEXURALNET_DNN_I_LAYER

#include "tensor.h"

namespace nexural {
	class ILayer {
	public:
		virtual ~ILayer() { }
		virtual void Setup(LayerShape& prevLayerShape) = 0;
		virtual void FeedForward(const Tensor& inputData) = 0;
		virtual Tensor* GetOutput() = 0;
		virtual Tensor* GetLayerErrors() = 0;
		virtual LayerShape GetOutputShape() = 0;
	};
}
#endif
