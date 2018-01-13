/* Copyright (C) 2017 Alexandru-Valentin Musat (contact@nexural.com)

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

#include "../src/nexuralnet/dnn.h"
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;

namespace nexural {
	PYBIND11_PLUGIN(nexuralnet) {

		NDArrayConverter::init_numpy();

		py::module m("nexuralnet", "neXt neural network");

		py::class_<Network>(m, "network")
			.def(py::init<const std::string &>())

			.def("run", (void (Network::*)(cv::Mat &)) &Network::Run, "Run the network")

			.def("deserialize", &Network::Deserialize, "Deserialize a trained network file")

			.def("saveFiltersImages", &Network::SaveFiltersImages, "Save filters images")

			.def("getResultJSON", &Network::GetResultJSON, "Get network result as JSON");


		py::class_<NetworkTrainer> networkTrainer(m, "trainer");
		networkTrainer.def(py::init<const std::string &, const std::string &>())

			.def("train", (void (NetworkTrainer::*)(const std::string &, const std::string &, const std::string &, const std::string &, NetworkTrainer::TrainingDataSource, NetworkTrainer::TargetDataSource)) &NetworkTrainer::Train, "Train the network")

			.def("serialize", &NetworkTrainer::Serialize, "Serialize a trained network file")

			.def("deserialize", &NetworkTrainer::Deserialize, "Deserialize a trained network file");;

		py::enum_<NetworkTrainer::TrainingDataSource>(networkTrainer, "trainingDataSource")
		    .value("IMAGES_DIRECTORY", NetworkTrainer::TrainingDataSource::IMAGES_DIRECTORY)
		    .value("TXT_DATA_FILE", NetworkTrainer::TrainingDataSource::TXT_DATA_FILE)
		    .value("MNIST_DATA_FILE", NetworkTrainer::TrainingDataSource::MNIST_DATA_FILE)
		    .export_values();

		py::enum_<NetworkTrainer::TargetDataSource>(networkTrainer, "targetDataSource")
		    .value("TXT_DATA_FILE", NetworkTrainer::TargetDataSource::TXT_DATA_FILE)
		    .value("MNIST_DATA_FILE", NetworkTrainer::TargetDataSource::MNIST_DATA_FILE)
		    .export_values();

		return m.ptr();
	}
}