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

#include "computational_base_layer.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER
#define _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER

namespace nexural {
	class ConvolutionalLayer : public ComputationalBaseLayer {
	public:
		ConvolutionalLayer(const LayerParams &layerParams) : ComputationalBaseLayer(layerParams) {
			_kernel_width = parser::ParseLong(_layerParams, "kernel_width");
			_kernel_height = parser::ParseLong(_layerParams, "kernel_height");
			_padding_width = parser::ParseLong(_layerParams, "padding_width");
			_padding_height = parser::ParseLong(_layerParams, "padding_height");
			_stride_width = parser::ParseLong(_layerParams, "stride_width");
			_stride_height = parser::ParseLong(_layerParams, "stride_height");
		}

		~ConvolutionalLayer() {

		}

		virtual void Setup(const LayerShape& prevLayerShape) {
			_inputShape.Resize(prevLayerShape);
			_outputShape.Resize(prevLayerShape);
			_outputData.Resize(_outputShape);
		}

		virtual void FeedForward(const Tensor& inputData) {

		}

		virtual void BackPropagate(const Tensor& layerErrors) {

		}

	private:
		Tensor _weights;
		long _kernel_width;
		long _kernel_height;
		long _padding_width;
		long _padding_height;
		long _stride_width;
		long _stride_height;
	};
}
#endif
