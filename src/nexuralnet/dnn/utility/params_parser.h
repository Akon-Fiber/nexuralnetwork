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

#include <map>
#include <string>
#include <vector>
#include "../data_types/general_data_types.h"

#ifndef NEXURALNET_UTILITY_DATA_PARSER_H
#define NEXURALNET_UTILITY_DATA_PARSER_H

namespace nexural {
	namespace parser
	{
		int ParseInt(std::map<std::string, std::string> &map, std::string key);
		int ParseLong(std::map<std::string, std::string> &map, std::string key);
		float_n ParseFloat(std::map<std::string, std::string> &map, std::string key);
		bool ParseBool(std::map<std::string, std::string> &map, std::string key);
		std::string ParseString(std::map<std::string, std::string> &map, std::string key);
		std::vector<std::string> ParseStrVector(std::map<std::string, std::string> &map, std::string key);
		std::vector<float_n> ParseFltVector(std::map<std::string, std::string> &map, std::string key);
	}
}
#endif
