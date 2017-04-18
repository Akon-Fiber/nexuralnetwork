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
#include <nexuralnet\dnn\layers\computational_layers\fully_connected_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, FULLY_CONNECTED_LAYER_TESTS)
{
	// Init data for testing
	Params layerParams { { "neurons", "5" } };
	LayerShape inputShape(1, 1, 2, 2);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(1, 1, 1, 5);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), prevLayerErrors(1, 1, 1, 5);
	FullyConnectedLayer fullyConnectedLayer(layerParams);
	fullyConnectedLayer.Setup(inputShape, 0);
	fullyConnectedLayer.SetupLayerForTraining();

	// Fill data for testing
	inputData.Fill({ 
		5, 9, 3, 6
	});

	feedForwardExpected.Fill({
		129, 194, 188, 294, 208
	});


	prevLayerErrors.Fill({
		6, 8, 9, 15, 3
	});

	layerErrorsExpected.Fill({
		387, 362, 443, 389
	});

	
	fullyConnectedLayer.SetWeights({
		3, 8, 5, 4, 9, 10, 13,
		3, 1, 11, 2, 12, 16, 8,
		18, 14, 16, 5, 7, 9
	});

	fullyConnectedLayer.SetBiases({
		3, 2, 6, 4, 8
	});

	// Running the code
	fullyConnectedLayer.FeedForward(inputData);
	feedForwardResult = fullyConnectedLayer.GetOutput();
	fullyConnectedLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = fullyConnectedLayer.GetLayerErrors();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
}
