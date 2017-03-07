// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_I_COMPUTATIONAL_LAYER
#define _NEXURALNET_DNN_I_COMPUTATIONAL_LAYER

#include "tensor.h"

namespace nexural {
	class IComputationalLayer {
	public:
		virtual ~IComputationalLayer() { }
		virtual void BackPropagate(const Tensor& layerErrors) = 0;
		virtual void Update() = 0;
	};
}
#endif
