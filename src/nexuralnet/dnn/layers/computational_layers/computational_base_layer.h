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

#include "../i_layer.h"
#include "i_computational_layer.h"

#ifndef NEXURALNET_DNN_LAYERS_COMPUTATIONAL_BASE_LAYER
#define NEXURALNET_DNN_LAYERS_COMPUTATIONAL_BASE_LAYER

namespace nexural {
	class ComputationalBaseLayer : public ILayer, public IComputationalLayer {
	public:
		ComputationalBaseLayer();
		ComputationalBaseLayer(const Params &layerParams);
		virtual ~ComputationalBaseLayer();

		virtual Tensor* GetOutput();
		virtual Tensor* GetLayerErrors();
		virtual LayerShape GetOutputShape();
		virtual Tensor* GetLayerWeights();
		virtual Tensor* GetLayerDWeights();
		virtual Tensor* GetLayerBiases();
		virtual Tensor* GetLayerDBiases();

		virtual bool HasWeights();
		virtual bool HasBiases();

		virtual std::string GetLayerID() const;
		virtual void Serialize(Serializer& serializer);
		virtual void Deserialize(Serializer& serializer);
		virtual void SetWeights(const std::vector<float_n>& values);
		virtual void SetBiases(const std::vector<float_n>& values);

	protected:
		Params _layerParams;
		LayerShape _inputShape;
		LayerShape _outputShape;
		Tensor _outputData;
		Tensor _layerErrors;
		Tensor _weights;
		Tensor _dWeights;
		Tensor _biases;
		Tensor _dBiases;
		bool _hasWeights;
		bool _hasBiases;
		std::string _layerID;
	};
	typedef std::shared_ptr<ComputationalBaseLayer> ComputationalBaseLayerPtr;
}
#endif 
