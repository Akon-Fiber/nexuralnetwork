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

#include "../stdafx.h"
#include <nexuralnet\dnn\layers\computational_layers\average_pooling_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, AVERAGE_POOLING_LAYER_TESTS)
{
	// Init data for testing
	Params layerParams { { "kernel_width", "2" }, { "kernel_height", "2" } };
	LayerShape inputShape(1, 1, 6, 6);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(1, 1, 3, 3);

	// Fill data for testing
	inputData.Fill({ 21, 5, 17, 35, 11, 19,
		2, 18, 4, 10, 9, 58,
		46, 38, 47, 18, 4, 6,
		90, 3, 13, 51, 2, 77,
		17, 14, 7, 1, 3, 5,
		10, 2, 12, 90, 81, 8 });
	feedForwardExpected.Fill({11.5, 16.5, 24.25, 
		44.25, 32.25, 22.25, 
		10.75, 27.5, 24.25});

	// Testing the code
	AveragePoolingLayer averagePoolingLayer(layerParams);
	averagePoolingLayer.Setup(inputShape, 0);
	averagePoolingLayer.FeedForward(inputData);
	feedForwardResult = averagePoolingLayer.GetOutput();
	
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
}
