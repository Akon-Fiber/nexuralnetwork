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

#include "network_info_writer.h"

namespace nexural {
	struct NetworkInfoWriter::impl {
		rapidjson::Document document;
	};

	NetworkInfoWriter::NetworkInfoWriter() : _impl(std::make_unique<impl>()) {
		_impl->document.SetObject();
	}

	NetworkInfoWriter::NetworkInfoWriter(const std::string dataPath) : _impl(std::make_unique<impl>()) {
		std::ifstream ifs(dataPath);
		rapidjson::IStreamWrapper isw(ifs);

		if (_impl->document.ParseStream(isw).HasParseError()) {
			throw std::runtime_error("The JSON source is not valid!");
		}
	}

	NetworkInfoWriter::~NetworkInfoWriter() {

	}

	void NetworkInfoWriter::AddParentNode(const std::string& parentNodeName) {
		_impl->document.AddMember(rapidjson::Value().SetString(parentNodeName.c_str(), _impl->document.GetAllocator()), rapidjson::Value().SetObject(), _impl->document.GetAllocator());
	}
	

	void NetworkInfoWriter::Save(const std::string& outputFilePath) {
		std::ofstream ofs(outputFilePath);
		rapidjson::OStreamWrapper osw(ofs);
		rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
		_impl->document.Accept(writer);
	}
}
