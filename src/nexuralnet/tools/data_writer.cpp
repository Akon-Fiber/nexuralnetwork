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

#include <experimental/filesystem>
#include <opencv2/highgui.hpp>
#include "converter.h"
namespace fs = std::experimental::filesystem;

#include "data_writer.h"

namespace nexural {
	namespace tools {
		void DataWriter::WriteTensorImages(const std::string& outputFolderPath, Tensor& tensor, const std::string baseImageName, 
			const bool channelsAsImage, const ImageExtension imgExtension) {
			if (!fs::is_directory(outputFolderPath)) {
				throw std::runtime_error("Specified directory for writing tensor images doesn't exists!");
			}

			std::vector<cv::Mat> images;
			converter::CvtTensorToVecOfMat(tensor, images, channelsAsImage);

			for (size_t idx = 0; idx < images.size(); idx++) {
				fs::path dir(outputFolderPath);
				fs::path file(baseImageName + "-" + std::to_string(idx) + GetExtension(imgExtension));
				fs::path fullPath = dir / file;
				cv::imwrite(fullPath.string(), images[idx]);
			}
		}

		std::string DataWriter::GetExtension(const ImageExtension imgExtension) {
			switch (imgExtension)
			{
			case ImageExtension::JPEG:
				return ".jpeg";
				break;
			case ImageExtension::JPG:
				return ".jpg";
				break;
			case ImageExtension::PNG:
				return ".png";
				break;
			case ImageExtension::BMP:
				return ".bmp";
				break;
			default:
				throw std::runtime_error("You need to specify a supported extension!");
				break;
			}
		}

	}
}
