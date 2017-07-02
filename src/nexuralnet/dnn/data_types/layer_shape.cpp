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

#include <vector>
#include "layer_shape.h"
#include "general_data_types.h"

namespace nexural {
	LayerShape::LayerShape() :
		_numSamples(0),
		_k(0),
		_nr(0),
		_nc(0) { }

	LayerShape::LayerShape(long numSamples, long k, long nr, long nc) :
		_numSamples(numSamples),
		_k(k),
		_nr(nr),
		_nc(nc) { }

	LayerShape::LayerShape(const LayerShape &other) {
		this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
	}

	LayerShape& LayerShape::operator=(const LayerShape other)
	{
		this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
		return *this;
	}

	bool LayerShape::operator==(const LayerShape& other) {
		return (this->_numSamples == other._numSamples &&
			this->_k == other._k &&
			this->_nr == other._nr &&
			this->_nc == other._nc);
	}

	bool LayerShape::operator!=(const LayerShape& other) {
		return !(*this == other);
	}

	void LayerShape::Resize(long numSamples, long k, long nr, long nc) {
		_numSamples = numSamples;
		_k = k;
		_nr = nr;
		_nc = nc;
	}

	void LayerShape::Resize(const LayerShape &other) {
		this->Resize(other.GetNumSamples(), other.GetK(), other.GetNR(), other.GetNC());
	}

	long LayerShape::Size() const { return _numSamples * _k * _nr * _nc; }

	void LayerShape::SetNumSamples(long numSamples) { _numSamples = numSamples; }
	void LayerShape::SetK(long k) { _k = k; }
	void LayerShape::SetNR(long nr) { _nr = nr; }
	void LayerShape::SetNC(long nc) { _nc = nc; }

	long LayerShape::GetNumSamples() const { return _numSamples; }
	long LayerShape::GetK() const { return _k; }
	long LayerShape::GetNR() const { return _nr; }
	long LayerShape::GetNC() const { return _nc; }
}