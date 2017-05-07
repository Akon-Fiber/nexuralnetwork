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
#include <nexuralnet/dnn/layers/computational_layers/relu_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, RELU_LAYER_TESTS)
{
	// Init layer
	Params layerParams;
	LayerShape inputShape(3, 2, 3, 2);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(3, 2, 3, 2);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), prevLayerErrors(3, 2, 3, 2);
	ReluLayer reluLayer(layerParams);
	reluLayer.Setup(inputShape, 0);
	reluLayer.SetupLayerForTraining();

	// Fill data for testing
	inputData.Fill({
		5, 9, -3, 6, -12, 7, 11, -9, 14, 21, 30, -42,
		0, 14, 22, 47, 35, 32, 0, 7, -29, 5, 33, 40,
		6, -17, 1, -24, 50, 3, -7, 9, 16, -20, 37, 0
	});

	feedForwardExpected.Fill({
		5, 9, 0, 6, 0, 7, 11, 0, 14, 21, 30, 0,
		0, 14, 22, 47, 35, 32, 0, 7, 0, 5, 33, 40,
		6, 0, 1, 0, 50, 3, 0, 9, 16, 0, 37, 0
	});

	prevLayerErrors.Fill({
		23, 19, -54, 15, -22, 35, 72, -91, 4, 15, 88, -2,
		10, 98, 12, 3, 25, 42, 0, 12, -91, 2, 9, 14,
		4, -37, 11, -14, 32, 30, -13, 49, 23, -40, 87, 10
	});

	layerErrorsExpected.Fill({
		23, 19, 0, 15, 0, 35, 72, 0, 4, 15, 88, 0,
		10, 98, 12, 3, 25, 42, 0, 12, 0, 2, 9, 14,
		4, 0, 11, 0, 32, 30, 0, 49, 23, 0, 87, 10
	});


	// Running the code
	reluLayer.FeedForward(inputData);
	feedForwardResult = reluLayer.GetOutput();
	reluLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = reluLayer.GetLayerErrors();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
}
