/* Copyright (C) 2017 Alexandru-Valentin Musat (contact@nexuralsoftware.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "../i_layer.h"
#include "i_loss_layer.h"

#ifndef NEXURALNET_DNN_LOSS_BASE_LAYER
#define NEXURALNET_DNN_LOSS_BASE_LAYER

namespace nexural {
	class LossBaseLayer : public ILayer, public ILossLayer {
	public:
		LossBaseLayer();
		LossBaseLayer(const Params& layerParams);
		virtual ~LossBaseLayer();
		virtual Tensor* GetOutput();
		virtual Tensor* GetLayerErrors();

		virtual LayerShape GetOutputShape();
		virtual const float_n GetTotalError();
		virtual const float_n GetPrecision();
		virtual const float_n GetRecall();
		virtual void ResetMetricsData();
		virtual const LayerShape GetTargetShape();
		virtual const NetworkResultType GetResultType();

	protected:
		// General layer params
		Params _layerParams;
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
		Tensor _layerErrors;
		NetworkResultType _resultType;

		// Params for training
		Tensor _confusionMatrix;
		float_n _totalError;
		float_n _precision;
		float_n _recall;
		size_t _numOfIterations;
	};
	typedef std::shared_ptr<LossBaseLayer> LossBaseLayerPtr;
}
#endif 
