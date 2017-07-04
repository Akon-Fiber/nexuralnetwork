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

#include "gray_image_input_layer.h"
#include "../../utility/params_parser.h"

namespace nexural {
	GrayImageInputLayer::GrayImageInputLayer(const Params &layerParams) : InputBaseLayer(layerParams) {
		long nr = parser::ParseLong(_layerParams, "input_height");
		long nc = parser::ParseLong(_layerParams, "input_width");
		_inputShape.Resize(1, 1, nr, nc);
		_outputShape.Resize(_inputShape);
		_outputData.Resize(_outputShape);
	}

	GrayImageInputLayer::~GrayImageInputLayer() {

	}

	void GrayImageInputLayer::LoadData(const Tensor& inputTensor) {
		if (inputTensor.GetShape() != _outputData.GetShape()) {
			throw std::runtime_error("Gray Input layer error: The input image is not of the same size as the layer!");
		}
		_outputData.ShareTensor(inputTensor);
	}
}
