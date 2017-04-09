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

#include "../data_types/layer_settings.h"
#include "json_utility/json_engine_reader.h"
#include "json_utility/json_config_reader.h"

#ifndef _NEXURALNET_UTILITY_CONFIG_READER_H
#define _NEXURALNET_UTILITY_CONFIG_READER_H

namespace nexural {
	enum class ConfigReaderType {
		JSON = 0
	};

	class ConfigReader {
	public:
		static void DecodeNetCongif(const std::string networkConfigPath, LayerSettingsCollection& layerSettingsCollection, const ConfigReaderType& readerType = ConfigReaderType::JSON) {
			if (readerType == ConfigReaderType::JSON) {
				JSONConfigReader jsonReader(networkConfigPath);
				jsonReader.DecodeNetConfigInternal(layerSettingsCollection);
			}
		}

		static void DecodeTrainerCongif(const std::string trainerConfigPath, Params& trainerParams, Params& solverParams, const ConfigReaderType& readerType = ConfigReaderType::JSON) {
			if (readerType == ConfigReaderType::JSON) {
				JSONConfigReader jsonReader(trainerConfigPath);
				jsonReader.DecodeTrainerConfigInternal(trainerParams, solverParams);
			}
		}
	};
}
#endif
