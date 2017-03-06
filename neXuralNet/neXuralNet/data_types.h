// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <map>

#ifndef _NEXURALNET_UTILITY_DATA_TYPES
#define _NEXURALNET_UTILITY_DATA_TYPES

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

		void Resize(long numSamples, long k, long nr, long nc) {
			_numSamples = numSamples;
			_k = k;
			_nr =nr;
			_nc = nc;
		}

		void SetNumSamples(long numSamples) { _numSamples = numSamples; }
		void SetK(long k) { _k = k; }
		void SetNR(long nr) { _nr = nr; }
		void SetNC(long nc) { _nc = nc; }

		long GetNumSamples() const { return _numSamples; }
		long GetK() const { return _k; }
		long GetNR() const { return _nr; }
		long GetNC() const { return _nc; }

	private:
		long _numSamples;
		long _k;
		long _nr;
		long _nc;
	};

}
#endif
