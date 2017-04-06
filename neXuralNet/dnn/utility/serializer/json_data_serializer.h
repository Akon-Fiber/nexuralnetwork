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

#include "base_serializer.h" 
#include "../json_utility/json_engine_reader.h"
#include <sstream>
#include <limits>
#include <iomanip>
#include <math.h>
#include <iostream>

#ifndef _NEXURALNET_UTILITY_JSON_DATA_SERIALIZER_H
#define _NEXURALNET_UTILITY_JSON_DATA_SERIALIZER_H

namespace nexural {
	class JSONSerializer : public JSONEngineReader, public BaseSerializer {
	public:
		JSONSerializer() : JSONEngineReader() {

		}

		JSONSerializer(const std::string dataPath, JSONSourceType sourceType = JSONSourceType::FROM_FILE) : JSONEngineReader(dataPath, sourceType) {

		}

		~JSONSerializer() {

		}

		void AddParentNode(const std::string& parentNodeName) {
			_document.AddMember(rapidjson::Value().SetString(parentNodeName.c_str(), _document.GetAllocator()), rapidjson::Value().SetObject(), _document.GetAllocator());
		}

		void SerializeTensor(const Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {
			rapidjson::Value tensorData(rapidjson::kObjectType);
			tensorData.AddMember("numSamples", tensor.GetNumSamples(), _document.GetAllocator());
			tensorData.AddMember("k", tensor.GetK(), _document.GetAllocator());
			tensorData.AddMember("nr", tensor.GetNR(), _document.GetAllocator());
			tensorData.AddMember("nc", tensor.GetNC(), _document.GetAllocator());

			//// Workaround: rapidjson array method fails, so use stringstream
			//std::stringstream ss;
			//ss << std::setprecision(std::numeric_limits<float>::digits10 + 1);

			//for (int i = 0; i < tensor.Size() - 1; i++) {
			//	if (std::isnan(tensor[i])) {
			//		std::cout << "tadam" << std::endl;
			//	}
			//	if (!(tensor[i] == tensor[i])) {
			//		std::cout << "nan" << std::endl;
			//	}
			//	if (tensor[i] <= DBL_MAX && tensor[i] >= -DBL_MAX) {
			//		std::cout << "inf" << std::endl;
			//	}

			//	ss << tensor[i] << ",";
			//}
			//ss << tensor[tensor.Size() - 1];
			//const std::string tmp = ss.str();
			//rapidjson::Value arrayData(tmp.c_str(), _document.GetAllocator());
			//tensorData.AddMember("host", arrayData.Move(), _document.GetAllocator());

			rapidjson::Value arrayData(rapidjson::kArrayType);
			for (int i = 0; i < tensor.Size(); i++) {
				arrayData.PushBack(rapidjson::Value().SetDouble(tensor[i]), _document.GetAllocator());
			}
			tensorData.AddMember("host", arrayData, _document.GetAllocator());

			rapidjson::Value jsonNodeName(nodeName.c_str(), _document.GetAllocator());
			_document[parentNodeName.c_str()].AddMember(jsonNodeName.Move(), tensorData, _document.GetAllocator());
		}

		void DeserializeTensor(Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {
			if (!_document.HasMember(parentNodeName.c_str())) {
				throw std::runtime_error(parentNodeName + " member is missing from the config file!");
			}

			const rapidjson::Value& node = _document[parentNodeName.c_str()];

			if (!node.HasMember(nodeName.c_str())) {
				throw std::runtime_error(nodeName + " member is missing from the config file!");
			}

			const rapidjson::Value& t = node[nodeName.c_str()];

			if (!t.HasMember("numSamples")) {
				throw std::runtime_error("numSamples member is missing from the config file!");
			}

			if (!t.HasMember("k")) {
				throw std::runtime_error("k member is missing from the config file!");
			}

			if (!t.HasMember("nr")) {
				throw std::runtime_error("nr member is missing from the config file!");
			}

			if (!t.HasMember("nc")) {
				throw std::runtime_error("nc member is missing from the config file!");
			}

			if (!t.HasMember("host")) {
				throw std::runtime_error("host member is missing from the config file!");
			}

			long numSamples = t["numSamples"].GetInt();
			long k = t["k"].GetInt();
			long nr = t["nr"].GetInt();
			long nc = t["nc"].GetInt();
			tensor.Resize(numSamples, k, nr, nc);

			/*std::string arrayData = t["host"].GetString();
			std::vector<std::string> tokens = helper::TokenizeString(arrayData, ",");

			for (long index = 0; index < tokens.size(); ++index)
			{
				tensor[index] = std::stof(tokens[index]);
			}*/

			const rapidjson::Value& arrayData = t["host"];
			long index = 0;
			for (rapidjson::Value::ConstValueIterator itr = arrayData.Begin(); itr != arrayData.End(); ++itr) {
				tensor[index] = static_cast<float_n>(itr->GetDouble());
				index++;
			}
		}

		void Save(const std::string& outputFilePath) {
			SerializeToFile(outputFilePath);
		}
	};
}
#endif
