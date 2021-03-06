cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(neXuralNetwork)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lstdc++fs")
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

find_package(OpenCV REQUIRED opencv_world)


file(GLOB_RECURSE nexuralnet_library_src
	"src/nexuralnet/*.h"
	"src/nexuralnet/*.cpp"
	)
add_library(nexural_static STATIC ${nexuralnet_library_src})
include_directories(${OpenCV_INCLUDE_DIRS})
set(RapidJSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rapidjson/include CACHE PATH "RapidSJSON include directory")
target_include_directories(nexural_static PUBLIC ${RapidJSON_INCLUDE_DIRS})
target_link_libraries(nexural_static ${OpenCV_LIBS})


file(GLOB_RECURSE nexuralnet_samples_src
	"samples/*.h"
	"samples/*.cpp"
	)
add_executable(samples ${nexuralnet_samples_src})
set(nexural_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "nexural include directory")
target_include_directories(samples PUBLIC ${nexural_INCLUDE_DIRS})
target_link_libraries(samples nexural_static ${OpenCV_LIBS})



add_subdirectory(thirdparty/pybind11)
find_package(PythonInterp)
find_package(NumPy)
pybind11_add_module(nexuralnet python/py_nexuralnet.cpp 
	thirdparty/pybind11_opencv_numpy/ndarray_converter.h 
	thirdparty/pybind11_opencv_numpy/ndarray_converter.cpp)
target_link_libraries(nexuralnet PRIVATE nexural_static ${OpenCV_LIBS})
target_include_directories(nexuralnet PUBLIC ${nexural_INCLUDE_DIRS})
target_include_directories(nexuralnet PRIVATE 
	./thirdparty/pybind11_opencv_numpy
	${PYTHON_NUMPY_INCLUDE_DIR})

install(TARGETS nexuralnet LIBRARY DESTINATION py_module)