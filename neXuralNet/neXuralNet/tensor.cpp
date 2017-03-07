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

namespace nexural {
	Tensor::Tensor() : 
		_numSamples(0),
		_k(0),
		_nr(0),
		_nc(0),
		_size(0)
	{
		_host.reset();
	}

	Tensor::Tensor(long numSamples_, long k_, long nr_, long nc_) :
		_numSamples(numSamples_),
		_k(k_),
		_nr(nr_),
		_nc(nc_) 
	{
		_size = numSamples_ * k_ * nr_ * nc_;
		_host.reset(new float[_size], std::default_delete<float[]>());
	}

	Tensor::~Tensor() { 
	}

	void Tensor::Resize(const long numSamples_, const long k_, const long nr_, const long nc_) {
		_numSamples = numSamples_;
		_k = k_;
		_nr = nr_;
		_nc = nc_;
		_size = _numSamples * _k * _nr * _nc;
		
		if (_size == 0) {
			_host.reset();
		} else {
			_host.reset(new float[_size], std::default_delete<float[]>());
		}
	}


	void Tensor::Resize(const LayerShape& layerShape) {
		_numSamples = layerShape.GetNumSamples();
		_k = layerShape.GetK();
		_nr = layerShape.GetNR();
		_nc = layerShape.GetNC();
		_size = _numSamples * _k * _nr * _nc;

		if (_size == 0) {
			_host.reset();
		}
		else {
			_host.reset(new float[_size], std::default_delete<float[]>());
		}
	}


	LayerShape Tensor::GetShape() {
		return LayerShape(_numSamples, _k, _nr, _nc);
	}
}