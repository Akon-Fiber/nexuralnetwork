// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_I_LOSS_LAYER
#define _NEXURALNET_DNN_I_LOSS_LAYER

#include "tensor.h"
#include "result_loss_base.h"

namespace nexural {

	class ILossLayer {
	public:
		virtual ~ILossLayer() { }
		virtual Tensor& GetOutput() = 0;
		virtual void FeedForward(const Tensor& inputData) = 0;
		virtual LossResultBasePtr GetResultType() = 0;
	};

}
#endif
