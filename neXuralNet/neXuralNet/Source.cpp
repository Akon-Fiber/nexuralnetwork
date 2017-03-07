// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "network.h"

int main(int argc, char* argv[]) {
	try {
		boost::filesystem::path full_path(boost::filesystem::initial_path<boost::filesystem::path>());
		full_path = boost::filesystem::system_complete(boost::filesystem::path(argv[0]));
		std::string configFilePath = full_path.parent_path().string() + "\\network.json";

		nexural::Network<nexural::OpenCVBGRImageLayer, cv::Mat> net(configFilePath);
		cv::Mat sourceImage = cv::imread(full_path.parent_path().parent_path().parent_path().parent_path().string() + "\\TestImages\\cat_8_gray.jpg");
		net.Run(sourceImage);
	}
	catch (std::exception stdEx) {
		std::cout << stdEx.what() << std::endl;
	}
	catch (cv::Exception cvEx) {
		std::cout << cvEx.what() << std::endl;
	}
	catch (...) {
		std::cout << "Something unexpected happened while running the network!" << std::endl;
	}
	return 0;
}