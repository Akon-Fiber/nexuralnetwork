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

#include <vector>
#include <map>

#ifndef _NEXURALNET_DATA_TYPES_LAYER_SHAPE_H
#define _NEXURALNET_DATA_TYPES_LAYER_SHAPE_H

namespace nexural {
	typedef std::map<std::string, std::string> LayerParams;

	struct LayerShape {
		LayerShape() :
			_numSamples(0),
			_k(0),
			_nr(0),
			_nc(0) { }

		LayerShape(long numSamples, long k, long nr, long nc) :
			_numSamples(numSamples),
			_k(k),
			_nr(nr),
			_nc(nc) { }

		LayerShape(const LayerShape &other) {
			this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
		}

		LayerShape& operator=(const LayerShape other)
		{
			this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
			return *this;
		}

		bool operator==(const LayerShape& other) {
			return (this->_numSamples == other._numSamples &&
				this->_k == other._k && 
				this->_nr == other._nr && 
				this->_nc == other._nc);
		}

		bool operator!=(const LayerShape& other) {
			return !(*this == other);
		}

		void Resize(long numSamples, long k, long nr, long nc) {
			_numSamples = numSamples;
			_k = k;
			_nr = nr;
			_nc = nc;
		}

		void Resize(const LayerShape &other) {
			this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
		}

		long Size() const { return _numSamples * _k * _nr * _nc; }

		void SetNumSamples(long numSamples) { _numSamples = numSamples; }
		void SetK(long k) { _k = k; }
		void SetNR(long nr) { _nr = nr; }
		void SetNC(long nc) { _nc = nc; }

		long GetNumSamples() const { return _numSamples; }
		long GetK() const { return _k; }
		long GetNR() const { return _nr; }
		long GetNC() const { return _nc; }

	private:
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
	};
}
#endif
