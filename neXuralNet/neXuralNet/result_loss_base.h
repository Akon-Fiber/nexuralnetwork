// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_RESULT_LOSS_BASE
#define _NEXURALNET_DNN_RESULT_LOSS_BASE

#include <memory>
#include "data_types.h"

namespace nexural {

	class LossResultBase {
	public:
		LossResultBase() { }
		virtual ~LossResultBase() { }
	};
	typedef std::shared_ptr<LossResultBase> LossResultBasePtr;
}
#endif 
