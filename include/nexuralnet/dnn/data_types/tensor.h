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
#include "layer_shape.h"

#ifndef _NEXURALNET_DATA_TYPES_TENSOR_H
#define _NEXURALNET_DATA_TYPES_TENSOR_H

namespace nexural {
	typedef double float_n;
	class Tensor {
		/*!
		WHAT THIS OBJECT REPRESENTS
		This object represents a 4D array of float values, all stored contiguously
		in memory.

		Finally, the convention is to interpret the tensor as a set of num_samples()
		3D arrays (images), each of dimension k() (channels) by nr() (rows) by nc() (columns).
		Also, while this class does not specify a memory layout, the convention is to
		assume that indexing into an element at coordinates (sample,k,nr,nc) can be
		accomplished via:
		host()[((sample*t.k() + k)*t.nr() + nr)*t.nc() + nc]

		THREAD SAFETY
		Instances of this object are not thread-safe.  So don't touch one from
		multiple threads at the same time.
		!*/
	public:
		Tensor();
		Tensor(long numSamples_, long k_, long nr_, long nc_);
		Tensor(const LayerShape& layerShape);
		~Tensor();
		explicit Tensor(const Tensor& tensor);
		Tensor& operator=(const Tensor& tensor);
		explicit Tensor(Tensor&& tensor);
		Tensor& operator=(Tensor&& tensor);

		typedef float_n* iterator;
		typedef const float_n* const_iterator;
		iterator       begin() { return  _host.get(); }
		const_iterator begin() const { return  _host.get(); }
		iterator       end() { return  _host.get() + _size; }
		const_iterator end() const { return  _host.get() + _size; }

		void Resize(const long numSamples_, const long k_, const long nr_, const long nc_);
		void Resize(const LayerShape& layerShape);
		void Reshape(const long numSamples_, const long k_, const long nr_, const long nc_);
		void Reshape(const LayerShape& layerShape);

		const float_n & operator [](long i) const { return _host.get()[i]; }
		float_n & operator [](long i) { return _host.get()[i]; }

		bool operator==(const Tensor& other) const;
		bool operator!=(const Tensor& other) const;

		void ShareTensor(const Tensor& tensor);
		void Clone(const Tensor& tensor);
		void Reset();

		long GetNumSamples() const { return _numSamples; }
		long GetK() const { return _k; }
		long GetNR() const { return _nr; }
		long GetNC() const { return _nc; }
		long Size() const { return _size; }

		void Fill(const float_n value);
		void Fill(const std::vector<float_n>& values);
		void FillRandom(const float_n range);
		void FillRandomBinomialDistribution();
		void GetBatch(const Tensor& tensor, const long startIndex, const long batchSize = 1);
		void GetShuffled(const Tensor& tensor, const std::vector<long>& tensorIndexes);
		LayerShape GetShape() const;

		void Flip180(const Tensor& tensor);

		void PrintToConsole() const;

	private:
		std::shared_ptr<float_n> _host;
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
		long _size;
	};
}
#endif
