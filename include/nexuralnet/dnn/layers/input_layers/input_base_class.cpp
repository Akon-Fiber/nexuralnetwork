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

#include "input_base_class.h"
#include "../../utility/params_parser.h"

namespace nexural {
	InputBaseLayer::InputBaseLayer() {

	}

	InputBaseLayer::InputBaseLayer(const Params &layerParams) {
		_layerParams = layerParams;
	}

	InputBaseLayer::~InputBaseLayer() {

	}

	void InputBaseLayer::SetInputBatchSize(const long batchSize) {
		_inputShape.Resize(batchSize, _inputShape.GetK(), _inputShape.GetNR(), _inputShape.GetNC());
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
	}

	Tensor* InputBaseLayer::GetOutput() {
		return &_outputData;
	}

	LayerShape InputBaseLayer::GetOutputShape() {
		return _outputShape;
	}
}
