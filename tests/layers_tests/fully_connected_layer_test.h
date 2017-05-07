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
#include <nexuralnet/dnn/layers/computational_layers/fully_connected_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, FULLY_CONNECTED_LAYER_TESTS)
{
	// Init layer
	Params layerParams{ { "neurons", "5" } };
	LayerShape inputShape(3, 2, 3, 2);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(3, 1, 1, 5);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), prevLayerErrors(3, 1, 1, 5);
	Tensor *dWeightsResult, dWeightsExpected(1, 1, 5, 12), *dBiasesResult, dBiasesExpected(1, 1, 1, 5);
	FullyConnectedLayer fullyConnectedLayer(layerParams);
	fullyConnectedLayer.Setup(inputShape, 0);
	fullyConnectedLayer.SetupLayerForTraining();

	// Set internal layer data
	fullyConnectedLayer.SetWeights({
		3, 8, 5, 4, 43, 15, 29, 61, 17, 4, 11, 22,
		9, 10, 13, 3, 30, 71, 15, 53, 31, 21, 18, 8,
		1, 11, 2, 12, 14, 10, 1, 3, 29, 37, 10, 7,
		12, 8, 18, 14, 5, 13, 4, 19, 26, 30, 48, 3,
		16, 5, 7, 9, 2, 17, 55, 27, 15, 20, 11, 5
	});

	fullyConnectedLayer.SetBiases({
		3, 2, 6, 4, 8
	});

	// Fill data for testing
	inputData.Fill({
		5, 9, 3, 6, 12, 7, 11, 9, 14, 21, 30, 42,
		18, 14, 22, 47, 35, 32, 11, 7, 29, 5, 33, 40,
		6, 17, 1, 24, 50, 3, 7, 9, 16, 20, 37, 3
	});

	feedForwardExpected.Fill({
		3194, 3444, 2241, 3200, 2369,
		4954, 6507, 3264, 4762, 3449,
		4030, 4212, 2848, 3855, 2253
	});

	prevLayerErrors.Fill({
		6, 8, 9, 15, 3,
		5, 18, 3, 7, 21,
		15, 13, 1, 30, 25
	});

	layerErrorsExpected.Fill({
		327, 362, 443, 393, 705, 994, 528, 1183, 1046, 1035, 1053, 319,
		600, 414, 538, 397, 874, 1831, 1601, 1968, 1227, 1139, 976, 401,
		923, 626, 961, 756, 1249, 1973, 2126, 2852, 1842, 1770, 2124, 656
	});

	dWeightsExpected.Fill({
		210, 379, 143, 631, 997, 247, 226, 224, 469, 451, 900, 497,
		442, 545, 433, 1206, 1376, 671, 377, 315, 842, 518, 1315, 1095,
		105, 140, 94, 219, 263, 162, 139, 111, 229, 224, 406, 501,
		381, 743, 229, 1139, 1925, 419, 452, 454, 893, 950, 1791, 1000,
		543, 746, 496, 1605, 2021, 768, 439, 399, 1051, 668, 1708, 1041
	});

	dBiasesExpected.Fill({
		26, 39, 13, 52, 49
	});


	// Running the code
	fullyConnectedLayer.FeedForward(inputData);
	feedForwardResult = fullyConnectedLayer.GetOutput();
	fullyConnectedLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = fullyConnectedLayer.GetLayerErrors();
	dWeightsResult = fullyConnectedLayer.GetLayerDWeights();
	dBiasesResult = fullyConnectedLayer.GetLayerDBiases();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
	ASSERT_EQ(*dWeightsResult, dWeightsExpected);
	ASSERT_EQ(*dBiasesResult, dBiasesExpected);
}
