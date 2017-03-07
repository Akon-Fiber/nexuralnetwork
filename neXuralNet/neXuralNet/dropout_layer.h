// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DNN_LAYERS_DROPOUT_LAYER
#define _NEXURALNET_DNN_LAYERS_DROPOUT_LAYER

#include "data_types.h"
#include "computational_base_layer.h"
#include "data_parser.h"
#include <iostream>
#include "utils.h"

namespace nexural {
	class DropoutLayer : public ComputationalBaseLayer {
	public:
		DropoutLayer(const LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {

		}

		~DropoutLayer() {

		}

		virtual void Setup(LayerShape& prevLayerShape) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(prevLayerShape);
			_outputData.Resize(_outputShape);
			_dropoutIndexes.Resize(prevLayerShape);
		}

		virtual void FeedForward(const Tensor& inputData) {
			Utils::RandomBinomialDistribution(_dropoutIndexes);

			for (long numSamples = 0; numSamples < inputData.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < inputData.GetK(); k++)
				{
					for (long nr = 0; nr < inputData.GetNR(); nr++)
					{
						for (long nc = 0; nc < inputData.GetNC(); nc++)
						{
							float value = inputData[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];
							int drop = _dropoutIndexes[(((numSamples * inputData.GetK()) + k) * inputData.GetNR() + nr) * inputData.GetNC() + nc];
							_outputData[(((numSamples * _outputData.GetK()) + k) * _outputData.GetNR() + nr) * _outputData.GetNC() + nc] = value * drop;
						}
					}
				}
			}
		}

		virtual void BackPropagate(const Tensor& layerErrors) {
			for (long numSamples = 0; numSamples < layerErrors.GetNumSamples(); numSamples++)
			{
				for (long k = 0; k < layerErrors.GetK(); k++)
				{
					for (long nr = 0; nr < layerErrors.GetNR(); nr++)
					{
						for (long nc = 0; nc < layerErrors.GetNC(); nc++)
						{
							float error = layerErrors[(((numSamples * layerErrors.GetK()) + k) * layerErrors.GetNR() + nr) * layerErrors.GetNC() + nc];
							int drop = _dropoutIndexes[(((numSamples * layerErrors.GetK()) + k) * layerErrors.GetNR() + nr) * layerErrors.GetNC() + nc];
							_layerErrors[(((numSamples * _layerErrors.GetK()) + k) * _layerErrors.GetNR() + nr) * _layerErrors.GetNC() + nc] = error * drop;
						}
					}
				}
			}
		}

		virtual void Update() {

		}

	private:
		Tensor _dropoutIndexes;
	};
}
#endif
