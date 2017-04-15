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
#include <rapidjson/filereadstream.h>
#include <fstream>

#include "json_config_reader.h"

namespace nexural {
	struct JSONConfigReader::impl {
		rapidjson::Document document;
	};

	JSONConfigReader::JSONConfigReader(const std::string networkConfigPath) : _impl(std::make_unique<impl>()) {
		std::ifstream ifs(networkConfigPath);
		rapidjson::IStreamWrapper isw(ifs);

		if (_impl->document.ParseStream(isw).HasParseError()) {
			throw std::runtime_error("The JSON source is not valid!");
		}
	}

	JSONConfigReader::~JSONConfigReader() {

	}

	void JSONConfigReader::DecodeNetConfigInternal(LayerSettingsCollection& layerSettingsCollection) {
		if (!_impl->document.HasMember("network_layers")) {
			throw std::runtime_error("network_layers member is missing from the config file!");
		}

		const rapidjson::Value& netLayers = _impl->document["network_layers"];

		for (rapidjson::SizeType i = 0; i < netLayers.Size(); i++)
		{
			const rapidjson::Value& currentLayer = netLayers[i];

			if (!currentLayer.HasMember("type")) {
				throw std::runtime_error("type member is missing from the config file!");
			}
			if (!currentLayer.HasMember("params")) {
				throw std::runtime_error("params member is missing from the config file!");
			}

			Params layerParams;
			std::string layerType = currentLayer["type"].GetString();

			const rapidjson::Value& paramsMember = currentLayer["params"];
			for (rapidjson::Value::ConstMemberIterator iter = paramsMember.MemberBegin(); iter != paramsMember.MemberEnd(); ++iter) {
				layerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
			}

			layerSettingsCollection.push_back(LayerSettings(layerType, layerParams));
		}
	}

	void JSONConfigReader::DecodeTrainerConfigInternal(Params& trainerParams, Params& solverParams) {
		if (!_impl->document.HasMember("trainer_settings")) {
			throw std::runtime_error("trainer_settings member is missing from the config file!");
		}

		if (!_impl->document.HasMember("solver")) {
			throw std::runtime_error("solver member is missing from the config file!");
		}

		const rapidjson::Value& trainerSettings = _impl->document["trainer_settings"];
		for (rapidjson::Value::ConstMemberIterator iter = trainerSettings.MemberBegin(); iter != trainerSettings.MemberEnd(); ++iter) {
			trainerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
		}

		const rapidjson::Value& solverMember = _impl->document["solver"];
		for (rapidjson::Value::ConstMemberIterator iter = solverMember.MemberBegin(); iter != solverMember.MemberEnd(); ++iter) {
			solverParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
		}
	}
}
