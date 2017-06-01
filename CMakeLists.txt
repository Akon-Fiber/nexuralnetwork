cmake_minimum_required(VERSION 3.7)

project(neXuralNet)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

file(GLOB_RECURSE nexuralnet_src
	"include/nexuralnet/*.h"
	"include/nexuralnet/*.cpp"
	)

find_package(OpenCV REQUIRED opencv_core opencv_imgproc opencv_highgui)

add_library(nexuralnet STATIC ${nexuralnet_src})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_objdet ${OpenCV_LIBS})

set(RapidJSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rapidjson/include CACHE PATH "RapidSJSON include directory")
target_include_directories(nexuralnet PUBLIC ${RapidJSON_INCLUDE_DIRS})

install(TARGETS nexuralnet
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)

#add_subdirectory(thirdparty/pybind11)
#pybind11_add_module(py_nexuralnet python/network.cpp)
#install(TARGETS py_nexuralnet LIBRARY DESTINATION bin)