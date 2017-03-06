// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <map>
#include "computational_base_layer.h"
#include "data_parser.h"

#ifndef _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER
#define _NEXURALNET_DNN_LAYERS_CONVOLUTIONAL_LAYER

namespace nexural {

	//class ConvolutionalLayer : public ComputationalBaseLayer {
	//public:
	//	ConvolutionalLayer(std::map<std::string, std::string> &layerParams, LayerShape input_shape) {
	//		_input_shape.width = input_shape.width;
	//		_input_shape.height = input_shape.height;

	//		_n_output_maps = parser::ParseInt(layerParams, "number_output_maps");
	//		_maps_width = parser::ParseInt(layerParams, "map_height");
	//		_maps_height = parser::ParseInt(layerParams, "map_height");

	//		_maps_width_stride = parser::ParseInt(layerParams, "map_width_stride");
	//		_maps_height_stride = parser::ParseInt(layerParams, "map_height_stride");

	//		_output_shape.width = input_shape.width - _maps_width + 1;
	//		_output_shape.height = input_shape.height - _maps_height + 1;

	//		for (int i = 0; i < _n_output_maps; i++) {
	//			Tensor t;
	//			//utils::generate_random_weights(_maps_width, _maps_height, 1, t);
	//			//_maps.push_back(t);
	//		}
	//		
	//	}

	//	~ConvolutionalLayer() {
	//		
	//	}
	//	
	//	virtual void Setup() {
	//	
	//	}
	//	
	//	virtual void FeedForward(Tensor inputData) {
	//		_convolve(inputData, _maps);
	//	}

	//	virtual void BackPropagate(Tensor layerErrors) {
	//	
	//	}

	//	virtual void Update() {

	//	}


	//private:
	//	void _convolve(Tensor& data, Tensor& maps) {

	//	}

	//private:
	//	float _l2_decay;
	//	int _n_output_maps;
	//	Tensor _maps;
	//	int _maps_width;
	//	int _maps_height;
	//	int _maps_width_stride;
	//	int _maps_height_stride;
	//	LayerShape _input_shape;
	//	LayerShape _output_shape;



	//};

}
#endif
