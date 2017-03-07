// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#ifndef _NEXURALNET_DATA_TYPES_TENSOR_H
#define _NEXURALNET_DATA_TYPES_TENSOR_H

#include <memory>
#include "data_types.h"

namespace nexural {
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
		explicit Tensor(const Tensor& t) = delete;
		explicit Tensor(Tensor&& t) = delete;

		Tensor& operator=(const Tensor&) = delete;
		Tensor& operator=(Tensor&& item) = delete;
		~Tensor();

		void Resize(const long numSamples_, const long k_, const long nr_, const long nc_);
		void Resize(const LayerShape& layerShape);

		const float operator [](long i) const { return _host.get()[i]; }
		float & operator [](long i) { return _host.get()[i]; }
		//float * operator [](int i) { return _host.get() + i; }

		long GetNumSamples() const { return _numSamples; }
		long GetK() const { return _k; }
		long GetNR() const { return _nr; }
		long GetNC() const { return _nc; }
		size_t Size() const { return _size; }

		LayerShape GetShape();

	private:
		std::shared_ptr<float> _host;
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
		long _size;
	};
}
#endif
