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

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <fstream>

#include "tensor.h"
#include "general_data_types.h"

#ifndef _NEXURALNET_UTILITY_JSON_ENGINE_READER_H
#define _NEXURALNET_UTILITY_JSON_ENGINE_READER_H

namespace nexural {
	class JSONEngineReader {
	public:
		enum class JSONSourceType {
			FROM_FILE = 0,
			FROM_STRING = 1
		};

		JSONEngineReader() { };

		JSONEngineReader(const std::string source, const JSONSourceType sourcetype) {
			if (sourcetype == JSONSourceType::FROM_FILE) {
				std::ifstream ifs(source);
				rapidjson::IStreamWrapper isw(ifs);

				if (_document.ParseStream(isw).HasParseError()) {
					throw std::runtime_error("The JSON source is not valid!");
				}
			}
			else if (sourcetype == JSONSourceType::FROM_STRING) {
				if (_document.Parse<0>(source.c_str()).HasParseError()) {
					throw std::runtime_error("The JSON source is not valid!");
				}
			}
		}

		virtual void GetMember(const std::string& memberName, std::string& value) {
			if (!_document.HasMember(memberName.c_str())) {
				throw std::runtime_error(memberName + " member is missing from the config file!");
			}

			const rapidjson::Value& netLayers = _document[memberName.c_str()];
			rapidjson::StringBuffer strbuf;
			rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
			netLayers.Accept(writer);
			std::string result(strbuf.GetString());
			value = result;
		}

		virtual void ToString(std::string& value) {
			rapidjson::StringBuffer strbuf;
			rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
			_document.Accept(writer);
			std::string result(strbuf.GetString());
			value = result;
		}

		virtual void SerializeToFile(const std::string outputFilePath) {
			std::ofstream ofs(outputFilePath);
			rapidjson::OStreamWrapper osw(ofs);
			rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
			_document.Accept(writer);
		}

		virtual ~JSONEngineReader() {

		}

	protected:
		rapidjson::Document _document;
	};
}
#endif
