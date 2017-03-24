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

#include "tensor.h"
#include "json_data_serializer.h"

#ifndef _NEXURALNET_UTILITY_DATA_SERIALIZER_H
#define _NEXURALNET_UTILITY_DATA_SERIALIZER_H

namespace nexural {
	enum class DataSerializationType {
		JSON = 0
	};

	class DataSerializer {
	public:
		static void SerializeTensor(const Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName, std::string& serializedData, const DataSerializationType& serializationType = DataSerializationType::JSON) {
			if (serializationType == DataSerializationType::JSON) {
				json_data_serializer::SerializeTensor(tensor, parentNodeName, nodeName, serializedData);
			}
		}

		static void DeserializeTensor(Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName, const std::string& serializedData, const DataSerializationType& serializationType = DataSerializationType::JSON) {
			if (serializationType == DataSerializationType::JSON) {
				json_data_serializer::DeserializeTensor(tensor, parentNodeName, nodeName, serializedData);
			}
		}

		static void Save(const std::string& filePath, const std::string& data, const DataSerializationType& serializationType = DataSerializationType::JSON) {
			if (serializationType == DataSerializationType::JSON) {
				json_data_serializer::Save(filePath, data);
			}
		}

		static void Load(const std::string& filePath, std::string& data, const DataSerializationType& serializationType = DataSerializationType::JSON) {
			if (serializationType == DataSerializationType::JSON) {
				json_data_serializer::Load(filePath, data);
			}
		}

		static void AddParentNode(const std::string& parentNodeName, std::string& data, const DataSerializationType& serializationType = DataSerializationType::JSON) {
			if (serializationType == DataSerializationType::JSON) {
				json_data_serializer::AddParentNode(parentNodeName, data);
			}
		}
	};
}
#endif
