cmake_minimum_required(VERSION 3.7)

project(neXuralNet)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

find_package(OpenCV REQUIRED opencv_world)


file(GLOB_RECURSE nexuralnet_library_src
	"include/nexuralnet/*.h"
	"include/nexuralnet/*.cpp"
	)
add_library(nexuralnet STATIC ${nexuralnet_library_src})
include_directories(${OpenCV_INCLUDE_DIRS})
set(RapidJSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rapidjson/include CACHE PATH "RapidSJSON include directory")
target_include_directories(nexuralnet PUBLIC ${RapidJSON_INCLUDE_DIRS})
target_link_libraries(nexuralnet ${OpenCV_LIBS})



file(GLOB_RECURSE nexuralnet_samples_src
	"samples/*.h"
	"samples/*.cpp"
	)
add_executable(samples ${nexuralnet_samples_src})
set(nexural_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "nexural include directory")
target_include_directories(samples PUBLIC ${nexural_INCLUDE_DIRS})
target_link_libraries(samples nexuralnet ${OpenCV_LIBS})



add_subdirectory(thirdparty/pybind11)
pybind11_add_module(py_nexuralnet python/network.cpp)
target_link_libraries(py_nexuralnet PRIVATE nexuralnet ${OpenCV_LIBS})
target_include_directories(py_nexuralnet PUBLIC ${nexural_INCLUDE_DIRS})

install(TARGETS py_nexuralnet LIBRARY DESTINATION bin)