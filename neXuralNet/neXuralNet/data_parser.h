// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include "exceptions.h"
#include "utils.h"

#ifndef _NEXURALNET_UTILITY_DATA_PARSER_H
#define _NEXURALNET_UTILITY_DATA_PARSER_H

namespace nexural {

	namespace parser
	{
		static int ParseInt(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			int value = -1;
			try
			{
				value = std::stoi(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamParseException));
			}

			return value;
		}

		static int ParseLong(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			int value = -1;
			try
			{
				value = std::stol(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamParseException));
			}

			return value;
		}

		static float ParseFloat(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			float value = -1;
			try
			{
				value = std::stof(map[key]);
			}
			catch (...)
			{
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamParseException));
			}

			return value;
		}

		static bool ParseBool(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			return map[key] == "1" || map[key] == "true";
		}

		static std::string ParseString(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			std::string value = map[key];

			return value;
		}

		static std::vector<std::string> ParseStrVector(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			std::vector<std::string> tokens = utils::tokenize_string(map[key], ",");
			return tokens;
		}

		static std::vector<float> ParseFltVector(std::map<std::string, std::string> &map, std::string key)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}


			std::vector<std::string> tokens = utils::tokenize_string(map[key], ",");

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
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamParseException));
			}

			return result;
		}


		static cv::Scalar ParseScalar(std::map<std::string, std::string> &map, std::string key, int channels_nb)
		{
			if (map.find(key) == map.end())
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamNotFound));

			if (map[key].empty()) {
				throw std::runtime_error(key + " " + enumToStr(mav_exception::EmptyParamValue));
			}

			cv::Scalar thresh(channels_nb);
			std::vector<std::string> tokens = utils::tokenize_string(map[key], ",");

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
				throw std::runtime_error(key + " " + enumToStr(mav_exception::ParamParseException));
			}

			return thresh;
		}
	}
}

#endif

