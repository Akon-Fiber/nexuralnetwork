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

#include "../stdafx.h"
#include <nexuralnet\dnn\layers\computational_layers\max_pooling_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, MAX_POOLING_LAYER_TESTS)
{
	// Init data for testing
	Params layerParams { { "kernel_width", "2" }, { "kernel_height", "2" } };
	LayerShape inputShape(3, 2, 6, 6);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(3, 2, 3, 3);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), prevLayerErrors(3, 2, 3, 3);

	// Fill data for testing
	inputData.Fill({ 
		21, 5, 17, 35, 11, 19,
		2, 18, 4, 10, 9, 58,
		46, 38, 47, 18, 4, 6,
		90, 3, 13, 51, 2, 77,
		17, 14, 7, 1, 3, 5,
		10, 2, 12, 90, 81, 8,
	
		13, 4, 73, 6, 80, 35,
		8, 3, 21, 17, 9, 85,
		7, 15, 10, 15, 3, 11,
		40, 60, 18, 33, 5, 65,
		8, 2, 50, 81, 7, 70,
		11, 7, 4, 24, 8, 1,
	
		7, 21, 9, 50, 31, 3,
		30, 3, 85, 4, 11, 17,
		6, 14, 12, 5, 27, 8,
		33, 23, 5, 90, 35, 40,
		19, 17, 40, 83, 8, 16,
		2, 13, 10, 80, 5, 1,
	
		8, 50, 15, 30, 40, 3, 
		20, 2, 7, 6, 16, 5,
		33, 5, 90, 41, 17, 73,
		14, 6, 1, 21, 3, 80,
		2, 60, 71, 3, 14, 19,
		18, 9, 4, 55, 38, 2,
	
		10, 8, 15, 29, 87, 96,
		17, 72, 37, 97, 83, 21,
		19, 42, 56, 2, 11, 5,
		4, 12, 1, 95, 3, 46, 
		32, 91, 51, 42, 6, 9,
		31, 40, 64, 25, 1, 3,
	
		26, 42, 91, 84, 1, 5,
		9, 56, 76, 48, 10, 25,
		15, 24, 93, 81, 62, 4,
		40, 19, 17, 3, 31, 97,
		46, 83, 62, 72, 83, 11,
		5, 11, 6, 51, 49, 53 });

	feedForwardExpected.Fill({
		21, 35, 58,
		90, 51, 77,
		17, 90, 81,
	
		13, 73, 85,
		60, 33, 65,
		11, 81, 70,
	
		30, 85, 31, 
		33, 90, 40,
		19, 83, 16,

		50, 30, 40,
		33, 90, 80,
		60, 71, 38,

		72, 97, 96,
		42, 95, 46,
		91, 64, 9,

		56, 91, 25,
		40, 93, 97,
		83, 72, 83 });


	prevLayerErrors.Fill({
		28.5, 91, 55,
		27, 44, 94,
		33, 56, 71,

		83, 62, 23,
		40, 34, 46,
		28, 56, 18,

		76, 31, 49,
		21, 93, 12,
		44, 6, 11,

		8, 74, 25,
		19, 31, 38,
		36, 98, 45,

		10, 56, 6,
		29, 43, 35,
		81, 18, 61,

		32, 24, 95,
		41, 56, 11,
		76, 48, 2 });

	layerErrorsExpected.Fill({
		28.5, 0, 0, 91, 0, 0,
		0, 0, 0, 0, 0, 55,
		0, 0, 0, 0, 0, 0,
		27, 0, 0, 44, 0, 94,
		33, 0, 0, 0, 0, 0,
		0, 0, 0, 56, 71, 0,

		83, 0, 62, 0, 0, 0,
		0, 0, 0, 0, 0, 23,
		0, 0, 0, 0, 0, 0,
		0, 40, 0, 34, 0, 46,
		0, 0, 0, 56, 0, 18,
		28, 0, 0, 0, 0, 0,

		0, 0, 0, 0, 49, 0,
		76, 0, 31, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		21, 0, 0, 93, 0, 12,
		44, 0, 0, 6, 0, 11,
		0, 0, 0, 0, 0, 0,

		0, 8, 0, 74, 25, 0,
		0, 0, 0, 0, 0, 0,
		19, 0, 31, 0, 0, 0,
		0, 0, 0, 0, 0, 38,
		0, 36, 98, 0, 0, 0,
		0, 0, 0, 0, 45, 0,

		0, 0, 0, 0, 0, 6,
		0, 10, 0, 56, 0, 0,
		0, 29, 0, 0, 0, 0,
		0, 0, 0, 43, 0, 35,
		0, 81, 0, 0 , 0, 61,
		0, 0, 18, 0, 0, 0,

		0, 0, 24, 0, 0, 0,
		0, 32, 0, 0, 0, 95,
		0, 0, 56, 0, 0, 0,
		41, 0, 0, 0, 0, 11,
		0, 76, 0, 48, 2, 0,
		0, 0, 0, 0, 0, 0 });

	// Running the code
	MaxPoolingLayer maxPoolingLayer(layerParams);
	maxPoolingLayer.Setup(inputShape, 0);
	maxPoolingLayer.SetupLayerForTraining();
	maxPoolingLayer.FeedForward(inputData);
	feedForwardResult = maxPoolingLayer.GetOutput();
	maxPoolingLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = maxPoolingLayer.GetLayerErrors();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
}
