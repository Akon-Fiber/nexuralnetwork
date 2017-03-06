// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <algorithm>
#include "data_types.h"

#ifndef _NEXURALNET_DNN_ACTIVATIONS_ACTIVATION_FUNCTIONS
#define _NEXURALNET_DNN_ACTIVATIONS_ACTIVATION_FUNCTIONS

namespace nexural {
	namespace activation {

		template <typename T> inline T sqr(T value) { return value*value; }

		class function {
		public:
			function() = default;
			function(const function &) = default;
			function(function &&) = default;
			function &operator =(const function &) = default;
			function &operator =(function &&) = default;
			virtual ~function() = default;

			virtual float f(const tensor& v, size_t index) const = 0;
			virtual float df(float y) const = 0;
		};


		class identity : public function {
		public:
			float f(const tensor& v, size_t i) const override { return v[i]; }
			float df(float /*y*/) const override { return float(1); }
		};


		class sigmoid : public function {
		public:
			float f(const tensor& v, size_t i) const override { return float(1) / (float(1) + std::exp(-v[i])); }
			float df(float y) const override { return y * (float(1) - y); }
		};


		class relu : public function {
		public:
			float f(const tensor& v, size_t i) const override { return std::max(float(0), v[i]); }
			float df(float y) const override { return y > float(0) ? float(1) : float(0); }
		};
		typedef relu rectified_linear; // for compatibility


		class leaky_relu : public function {
		public:
			float f(const tensor& v, size_t i) const override { return (v[i] > float(0)) ? v[i] : float(0.01) * v[i]; }
			float df(float y) const override { return y > float(0) ? float(1) : float(0.01); }
		};


		class elu : public function {
		public:
			float f(const tensor& v, size_t i) const override { return (v[i]<float(0) ? (exp(v[i]) - float(1)) : v[i]); }
			float df(float y) const override { return (y > float(0) ? float(1) : (float(1) + y)); }
		};


		class tan_h : public function {
		public:
			float f(const tensor& v, size_t i) const override { return std::tanh(v[i]); }
			float df(float y) const override { return float(1) - sqr(y); }
		};

	} 
} 
#endif
