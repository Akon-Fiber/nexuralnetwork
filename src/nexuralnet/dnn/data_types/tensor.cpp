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
		_host.reset(new float_n[_size], std::default_delete<float_n[]>());
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

	bool Tensor::operator==(const Tensor& other) const {
		if (!(this->_size == other._size)) {
			return false;
		}
		for (long index = 0; index < _size; index++) {
			if ((*this)[index] != other[index]) {
				return false;
			}
		}
		return true;
	}

	bool Tensor::operator!=(const Tensor& other) const {
		return !(*this == other);
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
			_host.reset(new float_n[tensor._size], std::default_delete<float_n[]>());

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
			_host.reset(new float_n[_size], std::default_delete<float_n[]>());
		}
	}


	void Tensor::Resize(const LayerShape& layerShape) {
		Resize(layerShape.GetNumSamples(), layerShape.GetK(), layerShape.GetNR(), layerShape.GetNC());
	}

	void Tensor::Reshape(const long numSamples_, const long k_, const long nr_, const long nc_) {
		long newSize = numSamples_ * k_ * nr_ * nc_;

		if (newSize != _size) {
			throw std::runtime_error("Can't reshape the tensor. The total size of the tensor should be the same.");
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

	void Tensor::Fill(const float_n value) {
		for (long i = 0; i < _size; i++) {
			_host.get()[i] = value;
		}
	}

	void Tensor::Fill(const std::vector<float_n>& values) {
		if (values.size() == _size) {
			for (long i = 0; i < _size; i++) {
				_host.get()[i] = values[i];
			}
		}
	}

	void Tensor::FillRandom(const float_n range) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(-range, range);
		std::generate(begin(), end(), [&]() { return dis(gen); });
	}

	void Tensor::FillRandomBinomialDistribution() {
		std::default_random_engine gen;
		std::binomial_distribution<int> dis(1, 0.5);
		std::generate(begin(), end(), [&]() { return dis(gen); });
	}

	void Tensor::GetBatch(const Tensor& tensor, const long startIndex, const long batchSize) {
		if (this != &tensor) {
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
		} else {
			throw std::runtime_error("Can't assign a mini-batch to the same tensor.");
		}
	}

	void Tensor::GetShuffled(const Tensor& tensor, const std::vector<long>& tensorIndexes) {
		if (this != &tensor) {
			Resize(tensor._numSamples, tensor._k, tensor._nr, tensor._nc);

			for (long numSample = 0; numSample < tensor._numSamples; numSample++) {
				for (long k = 0; k < tensor._k; k++)
				{
					for (long nr = 0; nr < tensor._nr; nr++)
					{
						for (long nc = 0; nc < tensor._nc; nc++)
						{
							_host.get()[(((numSample * tensor._k) + k) * tensor._nr + nr) * tensor._nc + nc] =
								tensor[(((tensorIndexes[numSample] * tensor._k) + k) * tensor._nr + nr) * tensor._nc + nc];
						}
					}
				}
			}
		} else {
			throw std::runtime_error("Can't shuffle to the same tensor.");
		}
	}

	LayerShape Tensor::GetShape() const {
		return LayerShape(_numSamples, _k, _nr, _nc);
	}

	void Tensor::Flip180(const Tensor& tensor) {
		if (this != &tensor) {
			this->Resize(tensor.GetShape());

			long totalNumSamples = tensor.GetNumSamples();
			long totalK = tensor.GetK();
			long totalNC = tensor.GetNC();
			long totalNR = tensor.GetNR();

			for (long numSample = 0; numSample < totalNumSamples; numSample++) {
				for (long k = 0; k < totalK; k++) {
					for (long nc = 0; nc < totalNC / 2; nc++)
					{
						for (long nr = 0; nr < totalNR; nr++) {
							long indexHost = (((numSample * tensor._k) + k) * tensor._nr + nr) * tensor._nc + nc;
							long indexTensor = (((numSample * tensor._k) + k) * tensor._nr + (totalNR - nr - 1)) * tensor._nc + (totalNC - nc - 1);

							float_n valueHost = tensor[indexHost];
							float_n valueTensor = tensor[indexTensor];

							_host.get()[indexHost] = valueTensor;
							_host.get()[indexTensor] = valueHost;
						}
					}

					if (totalNC & 1) {
						// TODO: the center of matrix is calculated twice
						for (long nc = 0; nc <= totalNR / 2; nc++) {
							long indexHost = (((numSample * tensor._k) + k) * tensor._nr + nc) * tensor._nc + (totalNC / 2);
							long indexTensor = (((numSample * tensor._k) + k) * tensor._nr + (totalNR - nc - 1)) * tensor._nc + (totalNC / 2);

							float_n valueHost = tensor[indexHost];
							float_n valueTensor = tensor[indexTensor];

							_host.get()[indexHost] = valueTensor;
							_host.get()[indexTensor] = valueHost;
						}
					}
				}
			}
		} else {
			throw std::runtime_error("Can't flip to the same tensor.");
		}
	}

	void Tensor::PrintToConsole() const {
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