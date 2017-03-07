// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_I_LOSS_LAYER
#define _NEXURALNET_DNN_I_LOSS_LAYER

#include "tensor.h"

namespace nexural {
	class ILossLayer {
	public:
		virtual ~ILossLayer() { }
		virtual void CalculateError() = 0;
	};
}
#endif
