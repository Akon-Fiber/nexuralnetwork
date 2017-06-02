# neXuralNetwork (neXt neural network) #
This toolbox has been written as a part of my bachelor degree project, but it started as my ambition to study machine learning and related algorithms. It contains the implementation of a convolitional neural network that it's written in C++.

## Compiling neXuralNet C++ on Windows ##
Firstly you need to install the following softwares:
 * [CMake](https://cmake.org/) 
 * [Visual Studio](https://www.visualstudio.com/vs/community/) - on Windows only Visual Studio 2015 and newer are supported since pybind11 relies on various C++11 language features that break older versions of Visual Studio;
 * Python - you need to have a Python distribution installed in your machine for building the python module (preferable [Anaconda](https://www.continuum.io/downloads));
 
 neXuralNet uses following libraries as its dependencies:
  * [OpenCV](https://github.com/opencv/opencv) - because this library is huge, it isn't included with this project (you need to download/build it)
  * RapidJSON - included as submodule (no action required)
  * pybind11 - included as submodule (no action required)
  * GTest - included as submodule (no action required)
 
 Type the following to compile the neXuralNet:
 ```
git clone --recursive https://github.com/nitnelav1101/nexuralnetwork.git
cd nexuralnetwork
mkdir build
cd build
SET OpenCV_DIR=<YOUR_PATH_HERE>
cmake -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config Release --target INSTALL
 ```

## License ##
neXuralNetwork (neXt neural network) is licensed under the MIT License. See the [LICENSE](LICENSE.md) for the specific language governing permissions and limitations under the License.
