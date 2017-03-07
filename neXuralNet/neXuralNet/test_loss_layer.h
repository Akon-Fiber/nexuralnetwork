// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <map>
#include "loss_base_layer.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_LOSS_LAYER
#define _NEXURALNET_DNN_LAYERS_LOSS_LAYER

namespace nexural {
	class TestLossLayer : public LossBaseLayer {
	public:
		TestLossLayer(const LayerParams& layerParams) : LossBaseLayer(layerParams) {
			
		}

		~TestLossLayer() {

		}

		virtual void Setup(LayerShape& prevLayerShape) {
			_inputShape.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), prevLayerShape.GetNR(), prevLayerShape.GetNC());
			_outputShape.Resize(prevLayerShape.GetNumSamples(), prevLayerShape.GetK(), prevLayerShape.GetNR(), prevLayerShape.GetNC());
			_outputData.Resize(_outputShape);

		}

		virtual void FeedForward(const Tensor& inputData) { 
			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < inputData.GetK(); k++)
				{
					for (long nr = 0; nr < inputData.GetNR(); nr++)
					{
						for (long nc = 0; nc < inputData.GetNC(); nc++)
						{
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];
						}
					}
				}
			}
		}

	private:

	};
}
#endif
