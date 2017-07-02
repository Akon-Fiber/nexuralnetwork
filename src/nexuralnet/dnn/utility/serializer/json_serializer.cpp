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

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <fstream>

#include "json_serializer.h"

namespace nexural {
	struct JSONSerializer::impl {
		rapidjson::Document document;
	};

	JSONSerializer::JSONSerializer() : _impl(std::make_unique<impl>()) {
		_impl->document.SetObject();
	}

	JSONSerializer::JSONSerializer(const std::string dataPath) : _impl(std::make_unique<impl>()) {
		std::ifstream ifs(dataPath);
		rapidjson::IStreamWrapper isw(ifs);

		if (_impl->document.ParseStream(isw).HasParseError()) {
			throw std::runtime_error("The JSON source is not valid!");
		}
	}

	JSONSerializer::~JSONSerializer() {

	}

	void JSONSerializer::AddParentNode(const std::string& parentNodeName) {
		_impl->document.AddMember(rapidjson::Value().SetString(parentNodeName.c_str(), _impl->document.GetAllocator()), rapidjson::Value().SetObject(), _impl->document.GetAllocator());
	}

	void JSONSerializer::SerializeTensor(const Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {
		rapidjson::Value tensorData(rapidjson::kObjectType);
		tensorData.AddMember("numSamples", tensor.GetNumSamples(), _impl->document.GetAllocator());
		tensorData.AddMember("k", tensor.GetK(), _impl->document.GetAllocator());
		tensorData.AddMember("nr", tensor.GetNR(), _impl->document.GetAllocator());
		tensorData.AddMember("nc", tensor.GetNC(), _impl->document.GetAllocator());

		rapidjson::Value arrayData(rapidjson::kArrayType);
		for (int i = 0; i < tensor.Size(); i++) {
			arrayData.PushBack(rapidjson::Value().SetDouble(tensor[i]), _impl->document.GetAllocator());
		}
		tensorData.AddMember("host", arrayData, _impl->document.GetAllocator());

		rapidjson::Value jsonNodeName(nodeName.c_str(), _impl->document.GetAllocator());
		_impl->document[parentNodeName.c_str()].AddMember(jsonNodeName.Move(), tensorData, _impl->document.GetAllocator());
	}

	void JSONSerializer::DeserializeTensor(Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {
		if (!_impl->document.HasMember(parentNodeName.c_str())) {
			throw std::runtime_error(parentNodeName + " member is missing from the config file!");
		}

		const rapidjson::Value& node = _impl->document[parentNodeName.c_str()];

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

		const rapidjson::Value& arrayData = t["host"];
		long index = 0;
		for (rapidjson::Value::ConstValueIterator itr = arrayData.Begin(); itr != arrayData.End(); ++itr) {
			tensor[index] = static_cast<float_n>(itr->GetDouble());
			index++;
		}
	}

	void JSONSerializer::Save(const std::string& outputFilePath) {
		std::ofstream ofs(outputFilePath);
		rapidjson::OStreamWrapper osw(ofs);
		rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
		_impl->document.Accept(writer);
	}
}
