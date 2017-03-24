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

#include "json_engine_reader.h"
#include "tensor.h"

#ifndef _NEXURALNET_UTILITY_JSON_DATA_SERIALIZER_H
#define _NEXURALNET_UTILITY_JSON_DATA_SERIALIZER_H

namespace nexural {
	namespace json_data_serializer {
		static void SerializeTensor(const Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName, std::string& serializedData) {
			rapidjson::Document document;
			if (document.Parse<0>(serializedData.c_str()).HasParseError()) {
				throw std::runtime_error("The JSON source is not valid!");
			}
			rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

			rapidjson::Value tensorData(rapidjson::kObjectType);
			tensorData.AddMember("numSamples", tensor.GetNumSamples(), allocator);
			tensorData.AddMember("k", tensor.GetK(), allocator);
			tensorData.AddMember("nr", tensor.GetNR(), allocator);
			tensorData.AddMember("nc", tensor.GetNC(), allocator);

			rapidjson::Value dataArray(rapidjson::kArrayType);
			for (int i = 0; i < tensor.Size(); i++) {
				dataArray.PushBack(rapidjson::Value().SetDouble(tensor[i]), allocator);
			}
			tensorData.AddMember("host", dataArray, allocator);

			rapidjson::Value jsonNodeName(nodeName.c_str(), allocator);
			document[parentNodeName.c_str()].AddMember(jsonNodeName.Move(), tensorData, allocator);
			rapidjson::StringBuffer strbuf;
			rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
			document.Accept(writer);

			std::string result(strbuf.GetString());
			serializedData = result;
		}

		static void DeserializeTensor(Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName, const std::string& serializedData) {
			rapidjson::Document document;
			if (document.Parse<0>(serializedData.c_str()).HasParseError()) {
				throw std::runtime_error("The JSON source is not valid!");
			}

			if (!document.HasMember(parentNodeName.c_str())) {
				throw std::runtime_error(parentNodeName + " member is missing from the config file!");
			}

			const rapidjson::Value& node = document[parentNodeName.c_str()];

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
			
			const rapidjson::Value& tensorData = t["host"];
			long index = 0;
			for (rapidjson::Value::ConstValueIterator itr = tensorData.Begin(); itr != tensorData.End(); ++itr) {
				tensor[index] = static_cast<float>(itr->GetDouble());
				index++;
			}
		}

		static void Save(const std::string& filePath, const std::string& data) {
			JSONEngineReader jsonEngine(data, JSONEngineReader::JSONSourceType::FROM_STRING);
			jsonEngine.SerializeToFile(filePath);
		}

		static void Load(const std::string& filePath, std::string& data) {
			JSONEngineReader jsonEngine(filePath, JSONEngineReader::JSONSourceType::FROM_FILE);
			jsonEngine.ToString(data);
		}

		static void AddParentNode(const std::string& parentNodeName, std::string& data) {
			if (data.empty()) {
				data = "{ }";
			}
			
			rapidjson::Document document;
			if (document.Parse<0>(data.c_str()).HasParseError()) {
				throw std::runtime_error("The JSON source is not valid!");
			}

			rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
			document.AddMember(rapidjson::Value().SetString(parentNodeName.c_str(), allocator), rapidjson::Value().SetObject(), allocator);
			rapidjson::StringBuffer strbuf;
			rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
			document.Accept(writer);

			std::string result(strbuf.GetString());
			data = result;
		}
	}
}
#endif
