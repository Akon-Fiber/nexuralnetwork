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

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include "utils.h"

#ifndef _NEXURALNET_UTILITY_DATA_PARSER_H
#define _NEXURALNET_UTILITY_DATA_PARSER_H

namespace nexural {
	namespace parser
	{
		static int ParseInt(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			int value = -1;
			try
			{
				value = std::stoi(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error("Error while parsing!");
			}

			return value;
		}

		static int ParseLong(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			int value = -1;
			try
			{
				value = std::stol(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error("Error while parsing!");
			}

			return value;
		}

		static float ParseFloat(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			float value = -1;
			try
			{
				value = std::stof(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error("Error while parsing!");
			}

			return value;
		}

		static bool ParseBool(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			return map[key] == "1" || map[key] == "true";
		}

		static std::string ParseString(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			std::string value = map[key];

			return value;
		}

		static std::vector<std::string> ParseStrVector(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			std::vector<std::string> tokens = Utils::TokenizeString(map[key], ",");
			return tokens;
		}

		static std::vector<float> ParseFltVector(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}


			std::vector<std::string> tokens = Utils::TokenizeString(map[key], ",");

			int tokens_cnt = (int)tokens.size();
			std::vector<float> result(tokens_cnt);

			bool safely_parsed = true;

			for (size_t token_id = 0; token_id < tokens_cnt; ++token_id)
			{
				float value = 0;
				try
				{
					value = std::stof(tokens[token_id]);
				}
				catch (...)
				{
					safely_parsed = false;
					break;
				}

				result[token_id] = value;
			}

			if (!safely_parsed) {
				throw std::runtime_error("Error while parsing!");
			}

			return result;
		}


		static cv::Scalar ParseScalar(std::map<std::string, std::string> &map, std::string key, int channels_nb)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error("The key was not found!");

			if (map[key].empty()) {
				throw std::runtime_error("Empty key value!");
			}

			cv::Scalar thresh(channels_nb);
			std::vector<std::string> tokens = Utils::TokenizeString(map[key], ",");

			bool safely_parsed = true;

			for (size_t token_id = 0; token_id < std::min((int)tokens.size(), channels_nb); ++token_id)
			{
				float value = 0;
				try
				{
					value = std::stof(tokens[token_id]);
				}
				catch (...)
				{
					safely_parsed = false;
					break;
				}

				thresh[static_cast<int>(token_id)] = value;
			}

			if (!safely_parsed) {
				throw std::runtime_error("Error while parsing!");
			}

			return thresh;
		}
	}
}
#endif
