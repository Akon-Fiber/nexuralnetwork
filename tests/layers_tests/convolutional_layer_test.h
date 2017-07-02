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
		12, 3, 13,

		12, 3, 1,
		2, 14, 5,
		11, 8, 4,

		4, 51, 12,
		1, 8, 7,
		3, 5, 12,

		5, 9, 6,
		1, 3, 22,
		2, 12, 3,

		8, 6, 11,
		2, 16, 7,
		2, 3, 13,

		4, 5, 30, 
		1, 9, 5,
		7, 5, 3,

		25, 4, 8,
		12, 13, 9,
		2, 3, 7,

		5, 12, 1,
		2, 5, 7,
		18, 9, 4,

		19, 7, 5,
		5, 9, 2,
		17, 5, 4,

		5, 7, 4,
		9, 18, 6,
		3, 13, 7,

		9, 21, 1,
		4, 16, 8,
		5, 11, 6,

		3, 9, 1,
		27, 12, 6,
		17, 3, 6,

		7, 4, 18,
		3, 7, 13,
		5, 13, 9
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
		7, 2, 10, 34,

		12, 4, 1, 6,
		7, 13, 41, 12,
		5, 7, 8, 14,
		6, 9, 14, 21,

		2, 3, 18, 1,
		36, 13, 4, 11,
		6, 5, 21, 16,
		9, 7, 31, 4,

		1, 13, 15, 1,
		2, 7, 4, 2,
		4, 12, 3, 23,
		9, 12, 1, 3
	});

	feedForwardExpected.Fill({
		2994, 2956,
		3652, 2875,

		3378, 2641,
		2638, 2187,

		2413, 2375,
		2640, 2213,

		3308, 2683,
		2697, 2224,

		3738, 3014,
		2641, 2915,

		2114, 2612,
		2117, 2933,
		
		2032, 2695,
		2363, 2341,
		
		2059, 2260,
		2067, 2597,
		
		1847, 1738,
		2346, 2409,

		3003, 2516,
		2193, 2956
	});

	prevLayerErrors.Fill({
		3, 5,
		10, 21,

		9, 12, 
		4, 36,

		44, 38,
		5, 8,

		25, 2,
		9, 13,

		2, 15,
		5, 8,

		8, 15,
		23, 14,

		3, 25,
		9, 6,

		4, 5,
		34, 12,

		7, 12,
		2, 34,

		8, 5,
		20, 17
	});

	layerErrorsExpected.Fill({
		630, 1289, 1156, 452,
		492, 2421, 2335, 661,
		927, 1762, 2431, 1193,
		321, 1087, 799, 411,
		
		720, 1215, 2564, 1364,
		550, 2269, 3588, 1369,
		1024, 1832, 1840, 837,
		355, 923, 773, 598,

		1329, 1667, 933, 674,
		1155, 2752, 2533, 1438,
		427, 1782, 2596, 1835,
		190, 694, 1077, 600,

		247, 913, 514, 112,
		997, 2191, 2314, 592,
		901, 2578, 2107, 874,
		487, 1426, 1322, 460,

		257, 832, 1791, 680,
		751, 2451, 3064, 1102,
		987, 2443, 1980, 765,
		800, 1697, 905, 374,

		326, 766, 738, 388,
		1560, 2106, 2528, 1557,
		758, 2489, 3257, 1281,
		468, 956, 1571, 675
	});

	dWeightsExpected.Fill({
		783, 1217, 1539,
		1112, 1449, 1219,
		936, 795, 1142,

		1855, 972, 570,
		1269, 1349, 1317,
		727, 1431, 1692,

		772, 944, 932,
		909, 848, 1071,
		817, 741, 1367,

		799, 1051, 1180,
		907, 1918, 963,
		1205, 719, 1128,

		1479, 1099, 497,
		1573, 1507, 944,
		799, 1672, 1629,

		789, 1120, 1155,
		911, 966, 1245,
		904, 794, 2184,

		1782, 3050, 2510,
		1020, 1044, 1380,
		2953, 1499, 1843,

		3089, 2005, 1333,
		2716, 1509, 1374,
		1486, 2544, 2849,

		565, 882, 1147,
		1928, 1461, 2049,
		2131, 1287, 1699,

		793, 2350, 1200,
		1133, 1231, 1259,
		1983, 894, 1537,

		1403, 1159, 801,
		1792, 1682, 1169,
		749, 1990, 1505,

		761, 751, 754,
		1522, 640, 1559,
		1472, 477, 1290,

		1000, 1373, 1221,
		692, 820, 1081,
		677, 847, 936,

		1625, 705, 700,
		990, 958, 985,
		696, 1370, 1247,

		465, 615, 675,
		687, 720, 1044,
		664, 663, 798
	});

	dBiasesExpected.Fill({
		99, 104, 150, 104, 80
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
	ASSERT_EQ(*layerErrorsResult, layerErrorsExpected);
	ASSERT_EQ(*dWeightsResult, dWeightsExpected);
	ASSERT_EQ(*dBiasesResult, dBiasesExpected);
}
