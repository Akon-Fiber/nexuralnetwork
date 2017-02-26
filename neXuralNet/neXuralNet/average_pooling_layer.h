// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef MAVNET_DNN_LAYERS_AVERAGE_POOLING_LAYER
#define MAVNET_DNN_LAYERS_AVERAGE_POOLING_LAYER

#include "data_types.h"
#include "computational_base_layer.h"
#include "data_parser.h"
#include <iostream>

namespace nexural {

	class AveragePoolingLayer : public ComputationalBaseLayer {
	public:
		AveragePoolingLayer(LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {

		}

		~AveragePoolingLayer() {
			
		}
		
		virtual void Setup() {

			_outputData.Resize(1, 1, parser::ParseLong(_layerParams, "kernel_width"), parser::ParseLong(_layerParams, "kernel_height"));
		}
		
		virtual void FeedForward(const Tensor& inputData) {
			_outputData.Resize(inputData.GetNumSamples(), inputData.GetK(), inputData.GetNR(), inputData.GetNC());

			for (long numSamples = 0; numSamples < _outputData.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < _outputData.GetK(); k++) {
					for (long nr = 0; nr < _outputData.GetNR(); nr++)
					{
						for (long nc = 0; nc < _outputData.GetNC(); nc++)
						{
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] =
								inputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] - 2;
						}
					}
				}
			}


			/*for (int i = 0; i < _outputData.Size(); i++) {
				std::cout << _outputData[i] << std::endl;
			}*/

		}

		virtual void BackPropagate(const Tensor& layerErrors) {
		
		}

		virtual void Update() {

		}


	private:
		
	};

}
#endif
