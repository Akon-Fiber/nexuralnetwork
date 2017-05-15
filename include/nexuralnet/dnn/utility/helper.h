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

#ifndef _NEXURALNET_UTILITY_HELPER_H
#define _NEXURALNET_UTILITY_HELPER_H

namespace nexural {
	namespace helper {
		static std::vector<std::string> TokenizeString(const std::string& str, const std::string& delimiters) {
			std::vector<std::string> tokens;
			std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
			std::string::size_type pos = str.find_first_of(delimiters, lastPos);

			while (std::string::npos != pos || std::string::npos != lastPos)
			{
				tokens.push_back(str.substr(lastPos, pos - lastPos));
				lastPos = str.find_first_not_of(delimiters, pos);
				pos = str.find_first_of(delimiters, lastPos);
			}
			return tokens;
		}

		// TODO: If the project is updated to use c++17, change with the new std::clamp method
		template <typename T>
		static T clip(const T& n, const T& lower, const T& upper) {
			return std::max(lower, std::min(n, upper));
		}

		static void SplitData(const Tensor& data, Tensor &firstFold, Tensor &secondFold, const float_n firstFoldPercentage = 90) {
			long dataNumOfSamples = data.GetNumSamples();
			long foldNumOfSamples = 0;
			long numberOfElementsPerSample = data.GetK() * data.GetNR() * data.GetNC();
			int separator = static_cast<int>(std::round(dataNumOfSamples * firstFoldPercentage / 100));

			firstFold.Resize(separator, data.GetK(), data.GetNR(), data.GetNC());
			secondFold.Resize(dataNumOfSamples - separator, data.GetK(), data.GetNR(), data.GetNC());

			for (long numSample = 0; numSample < separator; numSample++) {
				for (long index = 0; index < numberOfElementsPerSample; index++) {
					firstFold[(foldNumOfSamples * numberOfElementsPerSample) + index] = data[(numSample * numberOfElementsPerSample) + index];
				}
				foldNumOfSamples++;
			}

			foldNumOfSamples = 0;
			for (long numSample = separator; numSample < dataNumOfSamples; numSample++) {
				for (long index = 0; index < numberOfElementsPerSample; index++) {
					secondFold[(foldNumOfSamples * numberOfElementsPerSample) + index] = data[(numSample * numberOfElementsPerSample) + index];
				}
				foldNumOfSamples++;
			}
		}

		static void BestClassClassification(const Tensor& tensor, size_t& bestClass) {
			// TODO: Support multiple samples
			if (tensor.GetNumSamples() == 1) {

				long tensorSize = tensor.Size();
				if (tensorSize > 0) {
					float_n maxValue = tensor[0];
					size_t maxIndex = 0;

					for (long idx = 1; idx < tensorSize; idx++) {
						float_n value = tensor[idx];
						if (value > maxValue) {
							maxValue = value;
							maxIndex = idx;
						}
					}
					bestClass = maxIndex;
				}
			} else {
				throw std::runtime_error("Can't use best class finder for tensors with multiple samples!");
			}
		}
	}
}
#endif
