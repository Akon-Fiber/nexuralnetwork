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

#include "../../data_types/tensor.h"
#include "dnn_base_result.h"

#ifndef NEXURALNET_DNN_I_LOSS_LAYER
#define NEXURALNET_DNN_I_LOSS_LAYER

namespace nexural {
	class ILossLayer {
	public:
		virtual ~ILossLayer() { }
		virtual void Setup(const LayerShape& prevLayerShape) = 0;
		virtual void CalculateError(const Tensor& targetData) = 0;
		virtual void CalculateTrainingMetrics(const Tensor& targetData) = 0;
		virtual const float_n GetTotalError() = 0;
		virtual const float_n GetPrecision() = 0;
		virtual const float_n GetRecall() = 0;
		virtual void ResetMetricsData() = 0;
		virtual void SetResult() = 0;
		virtual DNNBaseResult* GetResult() = 0;
		virtual const LayerShape GetTargetShape() = 0;
		virtual const std::string GetResultJSON() = 0;
		virtual const std::string GetResultType() = 0;
	};
}
#endif
