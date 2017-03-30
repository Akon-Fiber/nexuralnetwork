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
#include <algorithm>
#include <random>
#include <iostream>

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

	Tensor::Tensor(const LayerShape& layerShape) : 
		Tensor(layerShape.GetNumSamples(), layerShape.GetK(), layerShape.GetNR(), layerShape.GetNC()) { }

	Tensor::Tensor(const Tensor& tensor) {
		this->Clone(tensor);
	}

	Tensor& Tensor::operator=(const Tensor& tensor) {
		this->Clone(tensor);
		return *this;
	}

	Tensor::Tensor(Tensor&& tensor) {
		if (this != &tensor) {
			_host.swap(tensor._host);
			std::swap(_numSamples, tensor._numSamples);
			std::swap(_k, tensor._k);
			std::swap(_nr, tensor._nr);
			std::swap(_nc, tensor._nc);
			std::swap(_size, tensor._size);
		}
	}

	Tensor& Tensor::operator=(Tensor&& tensor) {
		if (this != &tensor) {
			_host.swap(tensor._host);
			std::swap(_numSamples, tensor._numSamples);
			std::swap(_k, tensor._k);
			std::swap(_nr, tensor._nr);
			std::swap(_nc, tensor._nc);
			std::swap(_size, tensor._size);
		}
		return *this;
	}

	void Tensor::ShareTensor(const Tensor& tensor) {
		if (this != &tensor) {
			_host = tensor._host;
			_numSamples = tensor._numSamples;
			_k = tensor._k;
			_nr = tensor._nr;
			_nc = tensor._nc;
			_size = _numSamples * _k * _nr * _nc;
		}
	}

	void Tensor::Clone(const Tensor& tensor) {
		if (this != &tensor) {
			_host.reset(new float[tensor._size], std::default_delete<float[]>());

			for (int i = 0; i < tensor._size; i++) {
				_host.get()[i] = tensor[i];
			}

			_numSamples = tensor._numSamples;
			_k = tensor._k;
			_nr = tensor._nr;
			_nc = tensor._nc;
			_size = tensor._size;
		}
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
		Resize(layerShape.GetNumSamples(), layerShape.GetK(), layerShape.GetNR(), layerShape.GetNC());
	}

	void Tensor::Reshape(const long numSamples_, const long k_, const long nr_, const long nc_) {
		long newSize = numSamples_ * k_ * nr_ * nc_;

		if (newSize != _size) {
			throw std::runtime_error("Can't reshape the tensor");
		}
		
		_numSamples = numSamples_;
		_k = k_;
		_nr = nr_;
		_nc = nc_;
		_size = _numSamples * _k * _nr * _nc;
	}

	void Tensor::Reshape(const LayerShape& layerShape) {
		Reshape(layerShape.GetNumSamples(), layerShape.GetK(), layerShape.GetNR(), layerShape.GetNC());
	}

	void Tensor::Reset() {
		this->Resize(0, 0, 0, 0);
	}

	void Tensor::Fill(const float value) {
		for (int i = 0; i < _size; i++) {
			_host.get()[i] = value;
		}
	}

	void Tensor::FillRandom() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 1);
		std::generate(Begin(), End(), [&]() { return dis(gen); });
	}

	void Tensor::FillRandomBinomialDistribution() {
		std::default_random_engine gen;
		std::binomial_distribution<int> dis(1, 0.5);
		std::generate(Begin(), End(), [&]() { return dis(gen); });
	}

	void Tensor::GetBatch(const Tensor& tensor, const long startIndex, const long batchSize) {
		Resize(batchSize, tensor._k, tensor._nr, tensor._nc);

		long numSampleUpper = (startIndex + batchSize) < tensor._numSamples ? (startIndex + batchSize) : tensor._numSamples;
		long newNumSample = 0;

		for (long numSample = startIndex; numSample < numSampleUpper; numSample++) {
			for (long k = 0; k < tensor._k; k++)
			{
				for (long nr = 0; nr < tensor._nr; nr++)
				{
					for (long nc = 0; nc < tensor._nc; nc++)
					{
						_host.get()[(((newNumSample * tensor._k) + k) * tensor._nr + nr) * tensor._nc + nc] =
							tensor[(((numSample * tensor._k) + k) * tensor._nr + nr) * tensor._nc + nc];
					}
				}
			}
			newNumSample++;
		}
	}

	LayerShape Tensor::GetShape() const {
		return LayerShape(_numSamples, _k, _nr, _nc);
	}

	void Tensor::OutputToConsole() {
		std::cout.precision(8);
		for (long numSamples = 0; numSamples < _numSamples; numSamples++) {
			std::cout << "=========== Numsample: " << numSamples << " =============" << std::endl;
			for (long k = 0; k < _k; k++) {
				for (long nr = 0; nr < _nr; nr++) {
					for (long nc = 0; nc < _nc; nc++) {
						std::cout << _host.get()[(((numSamples * _k) + k) * _nr + nr) * _nc + nc] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << "------------------------------" << std::endl;
			}
			std::cout << "================================" << std::endl << std::endl;
		}
	}
}