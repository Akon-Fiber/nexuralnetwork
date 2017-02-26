// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <map>

#ifndef MAVNET_UTILITY_DATA_TYPES
#define MAVNET_UTILITY_DATA_TYPES

namespace nexural {

	typedef std::map<std::string, std::string> LayerParams;

	struct LayerShape {
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

		void SetNumSamples(long numSamples) { _numSamples = numSamples; }
		void SetK(long k) { _k = k; }
		void SetNR(long nr) { _nr = nr; }
		void SetNC(long nc) { _nc = nc; }

		long GetNumSamples() { return _numSamples; }
		long GetK() { return _k; }
		long GetNR() { return _nr; }
		long GetNC() { return _nc; }

	private:
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
	};

}
#endif
