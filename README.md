# neXuralNetwork (neXt neural network) #
This toolbox has been written as a part of my bachelor degree project and contains the implementation of a convolitional neural network that it's written in C++ and exported to Python as a module. neXuralNetwork is and always will be an open source project where anyone can contribute to it.

## Compiling neXuralNet on Windows ##
Firstly you need to install the following softwares:
 * [CMake](https://cmake.org/) (version 3.7.0 or newer)
 * [Visual Studio](https://www.visualstudio.com/vs/community/) - only Visual Studio 2015 and newer are supported since pybind11 relies on various C++11 language features that break older versions of Visual Studio;
 * Python - needed for building the python module (we recomand [Anaconda](https://www.continuum.io/downloads) environment);
 
 neXuralNet uses following libraries as its dependencies:
  * [OpenCV](https://github.com/opencv/opencv) (version 3.0.0 or newer) - because this library is huge, it isn't included with this project (you need to download/build it). For VS2017 you can download and use the OpenCV from [here](https://drive.google.com/open?id=0B4fA8oSTAEXCX1MyLXZ5VHVoUGM)
  * RapidJSON - included as submodule (no action required)
  * pybind11 - included as submodule (no action required)
  * GTest - included as submodule (no action required) - used only for tests
 
 Type the following to compile the neXuralNet Python module:
 ```
git clone --recursive https://github.com/nitnelav1101/nexuralnetwork.git
cd nexuralnetwork
mkdir build
cd build
SET OpenCV_DIR=<YOUR_PATH_HERE>
cmake -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config Release --target INSTALL
 ```
When building process is complete, the final .pyd file (Python module) will be located in `<PROJECT_ROOT_DIR>/build/install/py_module`. Copy the content of this folder to the Python's `site-packages` directory.


## Usage at a glance ##
This is a minimal Python script of running neXural module (in order to run this script, you need to install also the OpenCV and Numpy Python modules):
 ```
import nexuralnet
import numpy as np
import cv2

# Run the trainer
trainer = nexuralnet.trainer('network.json', 'trainer.json')
trainer.train('<TRAINING_DATA>', '<TARGET_DATA>', 'trained_file.json', '<INFO_FOLDER_PATH>', <TRAINING_DATA_SOURCE_TYPE>, <TARGET_DATA_SOURCE_TYPE>);

# Run the network
net = nexuralnet.network('network.json')
net.deserialize('c:/Users/Alexandru Musat/Google Drive/neXuralNet/nexural-data/mnist_softmax/mnist.json')

image = cv2.imread('image.jpg', 0) # 0 - read as a gray image, for color use 1
net.run(image)
net.saveFiltersImages('trained_file.json')
result = net.getResultJSON()
print(result)
 ```

## License ##
neXuralNetwork (neXt neural network) is licensed under the MIT License. See the [LICENSE](LICENSE.md) for the specific language governing permissions and limitations under the License.
