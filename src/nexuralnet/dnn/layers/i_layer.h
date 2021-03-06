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

#include "../data_types/tensor.h"
#include "../utility/helper.h"

#ifndef NEXURALNET_DNN_I_LAYER
#define NEXURALNET_DNN_I_LAYER

namespace nexural {
	class ILayer {
	public:
		virtual ~ILayer() { }
		virtual void FeedForward(const Tensor& inputData, const NetworkState networkState) = 0;
		virtual Tensor* GetOutput() = 0;
		virtual Tensor* GetLayerErrors() = 0;
		virtual LayerShape GetOutputShape() = 0;
		virtual void SetupLayerForTraining() = 0;
	};
}
#endif
