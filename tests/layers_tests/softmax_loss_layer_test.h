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
#include <nexuralnet/dnn/layers/loss_layers/softmax_loss_layer.h>
using namespace nexural;

TEST(LAYERS_TESTS, SOFTMAX_LOSS_LAYER_TESTS)
{
	// Init layer
	Params layerParams;
	LayerShape inputShape(2, 1, 1, 10);
	Tensor inputData(inputShape);
	Tensor *feedForwardResult, feedForwardExpected(2, 1, 1, 10);
	Tensor *layerErrorsResult, layerErrorsExpected(inputShape), targetData(2, 1, 1, 10);
	SoftmaxLossLayer softmaxLossLayer(layerParams);
	softmaxLossLayer.Setup(inputShape);
	softmaxLossLayer.SetupLayerForTraining();

	// Fill data for testing
	inputData.Fill({
		0.021, 0.016, 0.4, 0.154, 0.091, 0.21, 0.37, 0.52, 0.078, 0.06,
		0.33, 0.26, 0.34, 0.41, 0.55, 0.12, 0.17, 0.38, 0.704, 0.73
	});

	feedForwardExpected.Fill({
		0.08304418, 0.08262999, 0.12131285, 0.09485721, 0.08906556, 0.10032077, 0.11772751, 0.13677985, 0.0879152, 0.08634688,
		0.09148153, 0.08529681, 0.09240093, 0.09910075, 0.113993, 0.07415348, 0.07795541, 0.09617188, 0.1329718, 0.1364744
	});

	targetData.Fill({
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0
	});

	layerErrorsExpected.Fill({

	});


	// Running the code
	softmaxLossLayer.FeedForward(inputData);
	feedForwardResult = softmaxLossLayer.GetOutput();
	softmaxLossLayer.CalculateError(targetData);
	softmaxLossLayer.CalculateTotalError(targetData);

	layerErrorsResult = softmaxLossLayer.GetLayerErrors();
	float_n totalError = softmaxLossLayer.GetTotalError();

	// Testing the results
	//ASSERT_EQ(*feedForwardResult, feedForwardExpected);
	//ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
	//ASSERT_EQ(totalError, 32.2);
}
