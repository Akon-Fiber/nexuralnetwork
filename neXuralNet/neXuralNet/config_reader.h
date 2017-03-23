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

#include <memory>
#include <string>
#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"

#include "tensor.h"

#ifndef _NEXURALNET_UTILITY_CONFIG_READER_H
#define _NEXURALNET_UTILITY_CONFIG_READER_H

namespace nexural {
	struct LayerDetails {
		LayerDetails() = delete;
		LayerDetails(std::string layerType_, LayerParams layerParams_) :
			layerType(layerType_),
			layerParams(layerParams_) { }
		~LayerDetails() { }
		std::string layerType;
		LayerParams layerParams;
	};
	typedef std::vector<LayerDetails> NetworkLayers;
	typedef std::map<std::string, std::string> TrainerSettings;

	class ConfigReader {
	public:
		static void DecodeNetCongif(const std::string networkConfigPath, NetworkLayers& networkLayers) {
			JSONReader jsonReader(networkConfigPath);
			jsonReader.DecodeNetConfigInternal(networkLayers);
		}

		static void DecodeTrainerCongif(const std::string trainerConfigPath, TrainerSettings& trainerSettings) {
			JSONReader jsonReader(trainerConfigPath);
			jsonReader.DecodeTrainerConfigInternal(trainerSettings);
		}

	private:
		class JSONReader {
		public:
			JSONReader(const std::string networkConfigPath) {
				_fp = fopen(networkConfigPath.c_str(), "rb");
				if (!_fp) {
					throw std::runtime_error("Cannot load the config file!");
				}
				char readBuffer[65536];
				rapidjson::FileReadStream is(_fp, readBuffer, sizeof(readBuffer));
				_document.ParseStream(is);
			}

			~JSONReader() {
				if (_fp) {
					fclose(_fp);
				}
			}

			void DecodeNetConfigInternal(NetworkLayers& networkLayers) {
				if (!_document.HasMember("NetworkLayers")) {
					throw std::runtime_error("NetworkLayers member is missing from the config file!");
				}

				const rapidjson::Value& netLayers = _document["NetworkLayers"];

				for (rapidjson::SizeType i = 0; i < netLayers.Size(); i++)
				{
					const rapidjson::Value& currentLayer = netLayers[i];

					if (!currentLayer.HasMember("type")) {
						throw std::runtime_error("type member is missing from the config file!");
					}
					if (!currentLayer.HasMember("params")) {
						throw std::runtime_error("params member is missing from the config file!");
					}

					LayerParams layerParams;
					std::string layerType = currentLayer["type"].GetString();
					const rapidjson::Value& paramsMember = currentLayer["params"];

					for (rapidjson::Value::ConstMemberIterator iter = paramsMember.MemberBegin(); iter != paramsMember.MemberEnd(); ++iter) {
						layerParams.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
					}

					networkLayers.push_back(LayerDetails(layerType, layerParams));
				}
			}

			void DecodeTrainerConfigInternal(TrainerSettings& trainerSettings) {
				if (!_document.HasMember("TrainerSettings")) {
					throw std::runtime_error("TrainerSettings member is missing from the config file!");
				}

				const rapidjson::Value& trSettings = _document["TrainerSettings"];
				for (rapidjson::Value::ConstMemberIterator iter = trSettings.MemberBegin(); iter != trSettings.MemberEnd(); ++iter) {
					trainerSettings.insert(std::pair<std::string, std::string>(iter->name.GetString(), iter->value.GetString()));
				}
			}

		private:
			FILE* _fp;
			rapidjson::Document _document;
		};

	private:

	};
}
#endif
