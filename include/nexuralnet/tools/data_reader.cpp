/* Copyright (C) 2016-2017 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

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

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <experimental/filesystem>

#include "data_reader.h"
#include "converter.h"
#include <iostream>

namespace fs = std::experimental::filesystem;

namespace nexural {
	namespace tools {
		void DataReader::ReadMNISTData(const std::string& filename, Tensor& tensor, long limit) {
			std::ifstream file(filename, std::ios::binary);
			if (file.is_open())
			{
				int magicNumber = 0;
				int numberOfImages = 0;
				int nRows = 0;
				int nCols = 0;

				file.read((char*)&magicNumber, sizeof(magicNumber));
				magicNumber = ReverseInt(magicNumber);
				file.read((char*)&numberOfImages, sizeof(numberOfImages));
				numberOfImages = ReverseInt(numberOfImages);
				file.read((char*)&nRows, sizeof(nRows));
				nRows = ReverseInt(nRows);
				file.read((char*)&nCols, sizeof(nCols));
				nCols = ReverseInt(nCols);

				if (limit > numberOfImages) {
					throw std::runtime_error("The MNIST dataset limit was exceeded!");
				}

				tensor.Resize(limit, 1, nRows, nCols);

				for (int numSamples = 0; numSamples < limit; ++numSamples)
				{
					for (int nr = 0; nr < nRows; ++nr)
					{
						for (int nc = 0; nc < nCols; ++nc)
						{
							unsigned char temp = 0;
							file.read((char*)&temp, sizeof(temp));
							tensor[((numSamples * tensor.GetK()) * tensor.GetNR() + nr) * tensor.GetNC() + nc] = (float_n)temp / 255.0;
						}
					}
				}
			}
		}

		void DataReader::ReadMNISTLabels(const std::string& filename, Tensor& tensor, long limit, ReadMNISTLabelsType labelsType) {
			std::ifstream file(filename, std::ios::binary);
			if (file.is_open())
			{
				int magicNumber = 0;
				int numberOfLabels = 0;

				file.read((char*)&magicNumber, sizeof(magicNumber));
				magicNumber = ReverseInt(magicNumber);
				file.read((char*)&numberOfLabels, sizeof(numberOfLabels));
				numberOfLabels = ReverseInt(numberOfLabels);

				if (limit > numberOfLabels) {
					throw std::runtime_error("The MNIST dataset limit was exceeded!");
				}

				if (labelsType == ReadMNISTLabelsType::FOR_CLASSIFICATION) {
					tensor.Resize(limit, 1, 1, 10);

					for (int numSamples = 0; numSamples < limit; ++numSamples)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));

						for (int number = 0; number < 10; number++) {
							if (static_cast<int>(temp) == number) {
								tensor[numSamples * tensor.GetNC() + number] = 1;
							}
							else
							{
								tensor[numSamples * tensor.GetNC() + number] = 0;
							}
						}
					}
				}
				else if (labelsType == ReadMNISTLabelsType::FOR_REGRESSION)
				{
					tensor.Resize(limit, 1, 1, 1);

					for (int numSamples = 0; numSamples < limit; ++numSamples)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						tensor[numSamples] = static_cast<int>(temp);
					}
				}
			}
		}

		void DataReader::ReadImagesFromDirectory(const std::string directoryPath, Tensor& tensor, ReadImageType imagesType) {
			auto readType = imagesType == ReadImageType::COLOR ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
			std::vector<cv::Mat> images;

			// TODO: Optimize this code
			std::vector<cv::String> imageNames;
			std::vector<std::string> allowedExtensions = { ".jpg", ".png" };
			for (size_t i = 0; i < allowedExtensions.size(); i++) {
				std::vector<cv::String> imageNamesCurrentExtension;
				cv::glob(
					directoryPath + "*" + allowedExtensions[i],
					imageNamesCurrentExtension,
					true
				);
				imageNames.insert(
					imageNames.end(),
					imageNamesCurrentExtension.begin(),
					imageNamesCurrentExtension.end()
				);
			}
			for (size_t i = 0; i < imageNames.size(); i++) {
				cv::Mat image = cv::imread(imageNames[i], readType);
				images.push_back(image);
			}

			converter::CvtVecOfMatToTensor(images, tensor);
		}

		void DataReader::ReadTensorFromFile(const std::string filePath, Tensor& tensor) {
			std::ifstream file(filePath, std::ios::binary);
			if (file.fail()) {
				throw std::runtime_error("Can't open the file in order to read data!");
			}

			long index = 0, numSamples, k, nr, nc;
			file >> numSamples;
			file >> k;
			file >> nr;
			file >> nc;
			tensor.Resize(numSamples, k, nr, nc);
			long size = tensor.Size() - 1;

			while (!file.eof())
			{
				float_n value;
				file >> value;
				tensor[index] = value;
				if (size == index) {
					break;
				}
				else {
					index++;
				}
			}
		}

		void DataReader::ReadATTData(const std::string& directoryPath, Tensor& training, Tensor& labels) {
			size_t datasetFolders = 40;
			Tensor auxTensor, auxLabels;
			std::vector<cv::Mat> images;
			std::vector<cv::String> imageNames;
			auxLabels.Resize(400, 1, 1, datasetFolders);
			
			long numSamplesAuxLabels = 0;
			for (size_t folder = 0; folder < datasetFolders; folder++)
			{
				for (size_t image = 0; image < 10; image++) {
					for (size_t classImage = 0; classImage < datasetFolders; classImage++) {
						if (folder == classImage) {
							auxLabels[numSamplesAuxLabels * auxLabels.GetNC() + classImage] = 1;
						}
						else
						{
							auxLabels[numSamplesAuxLabels * auxLabels.GetNC() + classImage] = 0;
						}
					}
					numSamplesAuxLabels++;
				}
			}

			std::vector<cv::String> imageNamesCurrentExtension;
			for (size_t folders = 1; folders <= datasetFolders; folders++) {
				imageNamesCurrentExtension.clear();
					cv::glob(
						directoryPath + "s" + std::to_string(folders) + "/*.pgm",
						imageNamesCurrentExtension,
						true
					);
					imageNames.insert(
						imageNames.end(),
						imageNamesCurrentExtension.begin(),
						imageNamesCurrentExtension.end()
					);
			}
			

			for (size_t i = 0; i < imageNames.size(); i++) {
				cv::Mat image = cv::imread(imageNames[i], cv::IMREAD_GRAYSCALE);
				cv::Mat resized;
				cv::resize(image, resized, cv::Size(32, 32));
				images.push_back(resized);
			}

			converter::CvtVecOfMatToTensor(images, auxTensor);

			//auxTensor.PrintToConsole();

			for (long idx = 0; idx < auxTensor.Size(); idx++) {
				auxTensor[idx] = auxTensor[idx] / 255;
			}

			std::vector<long> tensorIndexes;
			for (size_t i = 0; i < auxTensor.GetNumSamples(); i++) {
				tensorIndexes.push_back(i);
			}
			std::random_shuffle(tensorIndexes.begin(), tensorIndexes.end());

			training.GetShuffled(auxTensor, tensorIndexes);
			labels.GetShuffled(auxLabels, tensorIndexes);

			
		}


		int DataReader::ReverseInt(int i)
		{
			unsigned char ch1, ch2, ch3, ch4;
			ch1 = i & 255;
			ch2 = (i >> 8) & 255;
			ch3 = (i >> 16) & 255;
			ch4 = (i >> 24) & 255;
			return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
		}
	}
}
