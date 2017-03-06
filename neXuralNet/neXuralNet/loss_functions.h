// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <algorithm>
#include <assert.h>
#include "data_types.h"
#include "tensor.h"


#ifndef _NEXURALNET_DNN_LOSS_LOSS_FUNCTIONS
#define _NEXURALNET_DNN_LOSS_LOSS_FUNCTIONS

namespace nexural {

	// mean-squared-error loss function for regression
	class mse {
	public:
		static float f(const Tensor& y, const Tensor& t) {
			assert(y.Size() == t.Size());
			float_t d = 0.0;

			for (long i = 0; i < y.Size(); ++i)
				d += (y[i] - t[i]) * (y[i] - t[i]);

			return d / y.Size();
		}

		static Tensor* df(const Tensor& y, const Tensor& t) {
			assert(y.Size() == t.Size());
			Tensor d(t.Size());
			float_t factor = float_t(2) / static_cast<float_t>(t.Size());

			for (long i = 0; i < y.Size(); ++i)
				d[i] = factor * (y[i] - t[i]);

			return &d;
		}
	};

	//// absolute loss function for regression
	//class absolute {
	//public:
	//	static float_t f(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		float_t d = float_t(0);

	//		for (serial_size_t i = 0; i < y.size(); ++i)
	//			d += std::abs(y[i] - t[i]);

	//		return d / y.size();
	//	}

	//	static tensor df(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		tensor d(t.size());
	//		float_t factor = float_t(1) / static_cast<float_t>(t.size());

	//		for (serial_size_t i = 0; i < y.size(); ++i) {
	//			float_t sign = y[i] - t[i];
	//			if (sign < 0.f)
	//				d[i] = -float_t(1) * factor;
	//			else if (sign > 0.f)
	//				d[i] = float_t(1) * factor;
	//			else
	//				d[i] = float_t(0);
	//		}

	//		return d;
	//	}
	//};

	//// absolute loss with epsilon range for regression
	//// epsilon range [-eps, eps] with eps = 1./fraction
	//template<int fraction>
	//class absolute_eps {
	//public:
	//	static float_t f(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		float_t d = float_t(0);
	//		const float_t eps = float_t(1) / fraction;

	//		for (serial_size_t i = 0; i < y.size(); ++i) {
	//			float_t diff = std::abs(y[i] - t[i]);
	//			if (diff > eps)
	//				d += diff;
	//		}
	//		return d / y.size();
	//	}

	//	static tensor df(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		tensor d(t.size());
	//		const float_t factor = float_t(1) / static_cast<float_t>(t.size());
	//		const float_t eps = float_t(1) / fraction;

	//		for (serial_size_t i = 0; i < y.size(); ++i) {
	//			float_t sign = y[i] - t[i];
	//			if (sign < -eps)
	//				d[i] = -float_t(1) * factor;
	//			else if (sign > eps)
	//				d[i] = float_t(1) * factor;
	//			else
	//				d[i] = 0.f;
	//		}
	//		return d;
	//	}
	//};

	//// cross-entropy loss function for (multiple independent) binary classifications
	//class cross_entropy {
	//public:
	//	static float_t f(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		float_t d = float_t(0);

	//		for (serial_size_t i = 0; i < y.size(); ++i)
	//			d += -t[i] * std::log(y[i]) - (float_t(1) - t[i]) * std::log(float_t(1) - y[i]);

	//		return d;
	//	}

	//	static tensor df(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		tensor d(t.size());

	//		for (serial_size_t i = 0; i < y.size(); ++i)
	//			d[i] = (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));

	//		return d;
	//	}
	//};

	//// cross-entropy loss function for multi-class classification
	//class cross_entropy_multiclass {
	//public:
	//	static float_t f(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		float_t d = 0.0;

	//		for (serial_size_t i = 0; i < y.size(); ++i)
	//			d += -t[i] * std::log(y[i]);

	//		return d;
	//	}

	//	static tensor df(const tensor& y, const tensor& t) {
	//		assert(y.size() == t.size());
	//		tensor d(t.size());

	//		for (serial_size_t i = 0; i < y.size(); ++i)
	//			d[i] = -t[i] / y[i];

	//		return d;
	//	}
	//};

}
#endif
