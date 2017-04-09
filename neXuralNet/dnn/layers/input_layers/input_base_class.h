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

#include "i_input_layer.h"
#include "../../utility/params_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_INPUT_BASE_LAYER
#define _NEXURALNET_DNN_LAYERS_INPUT_BASE_LAYER

namespace nexural {
	class InputBaseLayer : public IInputLayer {
	public:
		InputBaseLayer() {

		}

		InputBaseLayer(const Params &layerParams) {
			_layerParams = layerParams;
		}

		virtual ~InputBaseLayer() {

		}

		virtual Tensor* GetOutput() {
			return &_outputData;
		}

		virtual LayerShape GetOutputShape() {
			return _outputShape;
		}

	protected:
		Params _layerParams;
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
	};
	typedef std::shared_ptr<InputBaseLayer> InputBaseLayerPtr;
}
#endif 
