// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <string>
#include <algorithm>
#include <math.h> 
#include <random>
#include <opencv2/core/core.hpp>

#ifndef MAVNET_UTILITY_UTILS_H
#define MAVNET_UTILITY_UTILS_H

namespace nexural {

	namespace utils {

		static void generate_random_weights(int m, int n, int k, std::vector<float> &vec) {
			//int size = m * n * k;
			//float min = 0 - std::sqrt(1 / size);
			//float max = std::sqrt(1 / size);
			//using value_type = float;
			//// We use static in order to instantiate the random engine
			//// and the distribution once only.
			//// It may provoke some thread-safety issues.
			//static std::uniform_int_distribution<value_type> distribution(
			//	std::numeric_limits<value_type>::min(),
			//	std::numeric_limits<value_type>::max());
			//static std::default_random_engine generator;

			//std::vector<value_type> vec(size);
			//std::generate(vec.begin(), vec.end(), []() { return distribution(generator); });
		}

		static std::vector<std::string> tokenize_string(const std::string& str, const std::string& delimiters)
		{
			std::vector<std::string> tokens;
			// Skip delimiters at beginning.
			std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
			// Find first "non-delimiter".
			std::string::size_type pos = str.find_first_of(delimiters, lastPos);

			while (std::string::npos != pos || std::string::npos != lastPos)
			{  // Found a token, add it to the vector.
				tokens.push_back(str.substr(lastPos, pos - lastPos));
				// Skip delimiters.  Note the "not_of"
				lastPos = str.find_first_not_of(delimiters, pos);
				// Find next "non-delimiter"
				pos = str.find_first_of(delimiters, lastPos);
			}
			return tokens;
		}
	}
}

#endif
