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

#include <vector>
#include <string>
#include "../data_types/tensor.h"

#ifndef NEXURALNET_UTILITY_HELPER_H
#define NEXURALNET_UTILITY_HELPER_H

namespace nexural {
	namespace helper {
		std::vector<std::string> TokenizeString(const std::string& str, const std::string& delimiters);
		void SplitData(const Tensor& data, Tensor &firstFold, Tensor &secondFold, const float_n firstFoldPercentage = 90);
		void BestClassClassification(const Tensor& tensor, size_t& bestClass);
		float_n clip(const float_n& n, const float_n& lower, const float_n& upper);
	}
}
#endif
