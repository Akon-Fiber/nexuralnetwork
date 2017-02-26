// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

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

	void Tensor::Resize(long numSamples_, long k_, long nr_, long nc_) {
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

}