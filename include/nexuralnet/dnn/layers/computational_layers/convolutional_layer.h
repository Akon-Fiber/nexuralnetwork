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

#ifndef NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER
#define NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER

namespace nexural {
	class ConvolutionalLayer : public ComputationalBaseLayer {
	public:
		ConvolutionalLayer(const Params &layerParams);
		~ConvolutionalLayer();

		virtual void Setup(const LayerShape& prevLayerShape, const size_t layerIndex);
		virtual void FeedForward(const Tensor& inputData, const FeedForwardType feedForwardType = FeedForwardType::RUN);
		virtual void SetupLayerForTraining();
		virtual void BackPropagate(const Tensor& prevLayerErrors);
		virtual void Serialize(Serializer& serializer);
		virtual void Deserialize(Serializer& serializer);

	private:
		void AddPadding(const nexural::Tensor& input, nexural::Tensor& output, long paddingWidth, long paddingHeight);

	private:
		Tensor _internalInputData;
		Tensor _rotatedWeights;
		long _numOfFilters;
		long _kernelWidth;
		long _kernelHeight;
		long _paddingWidth;
		long _paddingHeight;
		long _strideWidth;
		long _strideHeight;
	};
}
#endif
