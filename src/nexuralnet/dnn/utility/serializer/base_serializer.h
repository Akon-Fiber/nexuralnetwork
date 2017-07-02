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

#include "../../data_types/tensor.h"

#ifndef NEXURALNET_UTILITY_BASE_SERIALIZER_H
#define NEXURALNET_UTILITY_BASE_SERIALIZER_H

namespace nexural {
	class BaseSerializer {
	public:
		BaseSerializer() {

		}

		virtual ~BaseSerializer() {

		}

		virtual void AddParentNode(const std::string& parentNodeName) {
			
		}

		virtual void SerializeTensor(const Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {
			
		}

		virtual void DeserializeTensor(Tensor& tensor, const std::string& parentNodeName, const std::string& nodeName) {

		}

		virtual void Save(const std::string& outputFilePath) {

		}
	};
	typedef std::shared_ptr<BaseSerializer> BaseSerializerPtr;
}
#endif 
