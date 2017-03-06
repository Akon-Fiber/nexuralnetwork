// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <map>
#include "loss_base_layer.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_LOSS_LAYER
#define _NEXURALNET_DNN_LAYERS_LOSS_LAYER

namespace nexural {

	class TestLossLayer : public LossBaseLayer {
	public:
		TestLossLayer(std::map<std::string, std::string> &layerParams, LayerShape input_shape) {
			

		}

		~TestLossLayer() {

		}

		virtual void FeedForward(const Tensor& inputData) { 
			_outputData = inputData;
		}


		virtual LossResultBasePtr GetResultType() {
			return NULL;
		}


	private:


	};

}
#endif
