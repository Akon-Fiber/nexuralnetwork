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

#ifndef NEXURALNET_DATA_TYPES_LAYER_SHAPE_H
#define NEXURALNET_DATA_TYPES_LAYER_SHAPE_H

namespace nexural {
	struct LayerShape {
		LayerShape();
		LayerShape(long numSamples, long k, long nr, long nc);
		LayerShape(const LayerShape &other);
		LayerShape& operator=(const LayerShape other);

		bool operator==(const LayerShape& other);
		bool operator!=(const LayerShape& other);
		void Resize(long numSamples, long k, long nr, long nc);
		void Resize(const LayerShape &other);
		long Size() const;

		void SetNumSamples(long numSamples);
		void SetK(long k);
		void SetNR(long nr);
		void SetNC(long nc);

		long GetNumSamples() const;
		long GetK() const;
		long GetNR() const;
		long GetNC() const;

	private:
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
	};
}
#endif
