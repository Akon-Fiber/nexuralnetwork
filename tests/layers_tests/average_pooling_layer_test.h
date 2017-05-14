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
		11.5, 16.5, 24.25, 
		44.25, 32.25, 22.25, 
		10.75, 27.5, 24.25,
	
		7, 29.25, 52.25,
		30.5, 19, 21,
		7, 39.75, 21.5,
	
		15.25, 37, 15.5,
		19, 28, 27.5,
		12.75, 53.25, 7.5,
	
		20, 14.5, 16,
		14.5, 38.25, 43.25,
		22.25, 33.25, 18.25,
	
		26.75, 44.5, 71.75,
		19.25, 38.5, 16.25,
		48.5, 45.5, 4.75,
	
		33.25, 74.75, 10.25,
		24.5, 48.5, 48.5,
		36.25, 47.75, 49 });


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
		7.125, 7.125, 22.75, 22.75, 13.75, 13.75,
		7.125, 7.125, 22.75, 22.75, 13.75, 13.75,
		6.75, 6.75, 11, 11, 23.5, 23.5,
		6.75, 6.75, 11, 11, 23.5, 23.5,
		8.25, 8.25, 14, 14, 17.75, 17.75,
		8.25, 8.25, 14, 14, 17.75, 17.75,

		20.75, 20.75, 15.5, 15.5, 5.75, 5.75,
		20.75, 20.75, 15.5, 15.5, 5.75, 5.75,
		10, 10, 8.5, 8.5, 11.5, 11.5,
		10, 10, 8.5, 8.5, 11.5, 11.5,
		7, 7, 14, 14, 4.5, 4.5,
		7, 7, 14, 14, 4.5, 4.5,

		19, 19, 7.75, 7.75, 12.25, 12.25,
		19, 19, 7.75, 7.75, 12.25, 12.25,
		5.25, 5.25, 23.25, 23.25, 3, 3,
		5.25, 5.25, 23.25, 23.25, 3, 3,
		11, 11, 1.5, 1.5, 2.75, 2.75,
		11, 11, 1.5, 1.5, 2.75, 2.75,

		2, 2, 18.5, 18.5, 6.25, 6.25,
		2, 2, 18.5, 18.5, 6.25, 6.25,
		4.75, 4.75, 7.75, 7.75, 9.5, 9.5,
		4.75, 4.75, 7.75, 7.75, 9.5, 9.5,
		9, 9, 24.5, 24.5, 11.25, 11.25,
		9, 9, 24.5, 24.5, 11.25, 11.25,

		2.5, 2.5, 14, 14, 1.5, 1.5,
		2.5, 2.5, 14, 14, 1.5, 1.5,
		7.25, 7.25, 10.75, 10.75, 8.75, 8.75,
		7.25, 7.25, 10.75, 10.75, 8.75, 8.75,
		20.25, 20.25, 4.5, 4.5, 15.25, 15.25,
		20.25, 20.25, 4.5, 4.5, 15.25, 15.25,

		8, 8, 6, 6, 23.75, 23.75,
		8, 8, 6, 6, 23.75, 23.75,
		10.25, 10.25, 14, 14, 2.75, 2.75,
		10.25, 10.25, 14, 14, 2.75, 2.75,
		19, 19, 12, 12, 0.5, 0.5,
		19, 19, 12, 12, 0.5, 0.5 });

	// Running the code
	AveragePoolingLayer averagePoolingLayer(layerParams);
	averagePoolingLayer.Setup(inputShape, 0);
	averagePoolingLayer.SetupLayerForTraining();
	averagePoolingLayer.FeedForward(inputData);
	feedForwardResult = averagePoolingLayer.GetOutput();
	averagePoolingLayer.BackPropagate(prevLayerErrors);
	layerErrorsResult = averagePoolingLayer.GetLayerErrors();

	// Testing the results
	ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
}
