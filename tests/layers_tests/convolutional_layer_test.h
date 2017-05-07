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
#include <nexuralnet/dnn/layers/computational_layers/convolutional_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, CONVOLUTIONAL_LAYER_TESTS)
{
	// Init layer
	Params layerParams{ { "num_of_filters", "5" },
		{ "kernel_width", "3" },
		{ "kernel_height", "3" },
		{ "padding_width", "0" },
		{ "padding_height", "0" },
		{ "stride_width", "1" },
		{ "stride_height", "1" } };
	LayerShape inputShape(2, 3, 4, 4);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(2, 5, 2, 2);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), prevLayerErrors(2, 5, 2, 2);
	Tensor *dWeightsResult, dWeightsExpected(5, 3, 3, 3), *dBiasesResult, dBiasesExpected(1, 1, 1, 5);
	ConvolutionalLayer convolutionalLayer(layerParams);
	convolutionalLayer.Setup(inputShape, 0);
	convolutionalLayer.SetupLayerForTraining();

	// Set internal layer data
	convolutionalLayer.SetWeights({
		9, 6, 1,
		20, 11, 7,
		8, 5, 3,

		9, 5, 11,
		2, 18, 5,
		7, 15, 2,

		15, 9, 4,
		4, 31, 13,
		12, 3, 13
	});

	convolutionalLayer.SetBiases({
		4, 7, 5, 9, 11
	});

	// Fill data for testing
	inputData.Fill({
		2, 31, 18, 1,
		6, 3, 4, 11,
		51, 5, 18, 4,
		16, 9, 1, 8,

		12, 23, 8, 16,
		36, 13, 4, 1,
		9, 15, 27, 6,
		5, 7, 11, 14,

		1, 3, 5, 12,
		26, 7, 14, 22,
		31, 5, 13, 16,
		7, 2, 10, 34
	});

	feedForwardExpected.Fill({
		2994, 2956,
		3652, 2875
	});

	prevLayerErrors.Fill({
		
	});

	layerErrorsExpected.Fill({
		
	});

	dWeightsExpected.Fill({
		
	});

	dBiasesExpected.Fill({
		
	});


	// Running the code
	convolutionalLayer.FeedForward(inputData);
	feedForwardResult = convolutionalLayer.GetOutput();
	convolutionalLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = convolutionalLayer.GetLayerErrors();
	dWeightsResult = convolutionalLayer.GetLayerDWeights();
	dBiasesResult = convolutionalLayer.GetLayerDBiases();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	//ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
	//ASSERT_EQ(*dWeightsResult, dWeightsExpected);
	//ASSERT_EQ(*dBiasesResult, dBiasesExpected);
}
