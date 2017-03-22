/* Copyright (C) 2016-2017 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

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

#include "tensor.h"
#include "data_parser.h"
#include "input_base_class.h"

#ifndef _NEXURALNET_DNN_LAYERS_TENSOR_INPUT_LAYER
#define _NEXURALNET_DNN_LAYERS_TENSOR_INPUT_LAYER

namespace nexural {
	class TensorInputLayer : public InputBaseLayer {
	public:
		TensorInputLayer(const LayerParams &layerParams) : InputBaseLayer(layerParams) {
			long numSamples = parser::ParseLong(_layerParams, "num_samples");
			long k = parser::ParseLong(_layerParams, "k");
			long nr = parser::ParseLong(_layerParams, "nr");
			long nc = parser::ParseLong(_layerParams, "nc");
			_inputShape.Resize(numSamples, k, nr, nc);
			_outputShape.Resize(_inputShape);
			_outputData.Resize(_outputShape);
		}

		~TensorInputLayer() {

		}

		void LoadData(const Tensor& inputTensor) {
			if (inputTensor.GetShape() != _outputData.GetShape()) {
				throw std::runtime_error("The input tensor is not of the same size as the layer!");
			}
			_outputData.ShareTensor(inputTensor);
		}

	private:

	};
}
#endif
