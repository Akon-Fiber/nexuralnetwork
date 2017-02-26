// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include "network.h"
#include "input_layer.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

int main() {
	nexural::Network net("E:\\RESEARCH\\neXuralNet\\neXuralNet\\network.json");
	cv::Mat sourceImage = cv::imread("e:\\POZE\\Avatare\\suit.jpg");
	net.Run(sourceImage);
	return 0;
}