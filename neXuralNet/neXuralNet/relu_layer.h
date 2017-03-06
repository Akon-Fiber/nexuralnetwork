// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LAYERS_RELU_LAYER
#define _NEXURALNET_DNN_LAYERS_RELU_LAYER

#include "data_types.h"
#include "computational_base_layer.h"
#include "data_parser.h"
#include <iostream>

namespace nexural {

	class ReluLayer : public ComputationalBaseLayer {
	public:
		ReluLayer(LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {

		}

		~ReluLayer() {

		}

		virtual void Setup(LayerShape& prevLayerShape) {
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
							float value = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];

							if(value < 0){
								_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = value * 0.0;
							} else {
								_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = value;
							}

						}
					}
				}
			}

		}

		virtual void BackPropagate(const Tensor& layerErrors) {
			_layerErrors.Resize(layerErrors.GetNumSamples(), layerErrors.GetK(), layerErrors.GetNR(), layerErrors.GetNC());

			for (long numSamples = 0; numSamples < layerErrors.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < layerErrors.GetK(); k++)
				{
					for (long nr = 0; nr < layerErrors.GetNR(); nr++)
					{
						for (long nc = 0; nc < layerErrors.GetNC(); nc++)
						{
							float error = layerErrors[(((numSamples * layerErrors.GetK()) + k) * layerErrors.GetNR() + nr) * layerErrors.GetNC() + nc];

							if (error < 0) {
								_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc] = error * 0.0;
							}
							else {
								_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc] = error;
							}

						}
					}
				}
			}
		}

		virtual void Update() {

		}


	private:

	};

}
#endif
