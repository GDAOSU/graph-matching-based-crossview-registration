cmake_minimum_required(VERSION 2.8)
project(CrossViewRegistration)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

ADD_DEFINITIONS(-DGLOG_NO_ABBREVIATED_SEVERITIES)
ADD_DEFINITIONS(-DNOMINMAX)
ADD_DEFINITIONS(-DUSE_OPENMP)
ADD_DEFINITIONS(-DUSE_MATH_DEFINES)

if(WIN32)
    add_definitions(/DGOOGLE_GLOG_DLL_DECL=)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(TEST_ROOT_FOLDER "test")
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})


# OpenMP
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	ADD_DEFINITIONS(-DUSE_OPENMP)
else()
	message("-- Can't find OpenMP. Continuing without it.")
endif()

# OpenCV
FIND_PACKAGE(OpenCV)
if(OpenCV_FOUND)
	INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
	ADD_DEFINITIONS(${OpenCV_DEFINITIONS})
	if (OpenCV_VERSION VERSION_LESS "3.0.0")	    
		ADD_DEFINITIONS(-DOPENCV_IS_V2)
		MESSAGE(STATUS "OpenCV ${OpenCV_VERSION} found (include: ${OpenCV_INCLUDE_DIRS})")
	endif()
	MESSAGE(STATUS "OpenCV ${OpenCV_VERSION} found (include: ${OpenCV_INCLUDE_DIRS})")
else()
	MESSAGE("-- Can't find OpenCV. Please specify OpenCV directory using OpenCV_DIR variable")
endif()

# Ceres
find_package(Ceres REQUIRED COMPONENTS OpenMP)

# Eigen3
FIND_PACKAGE(Eigen3 ${SYSTEM_PACKAGE_REQUIRED})
if(EIGEN_FOUND)
	INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIRS})
	ADD_DEFINITIONS(${EIGEN_DEFINITIONS} -D_USE_EIGEN)
	SET(_USE_EIGEN TRUE)
endif()

# Generate lib for current project
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/src/alignment
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
)

FILE(GLOB Utils_HDRS  "src/utils/*.h" "src/utils/*.hpp")
FILE(GLOB Utils_SRCS  "src/utils/*.c" "src/utils/*.cc" "src/utils/*.cpp")
source_group(utils FILES ${Utils_HDRS} ${Utils_SRCS})

FILE(GLOB Alignment_HDRS  "src/alignment/*.h" "src/alignment/*.hpp")
FILE(GLOB Alignment_SRCS  "src/alignment/*.c" "src/alignment/*.cc" "src/alignment/*.cpp")
source_group(alignment FILES ${Alignment_HDRS} ${Alignment_SRCS})

SET(Basic_Files  
src/basic_structs.h
src/camera.h src/camera.cc
src/optimizer.h src/optimizer.cc
src/structure.h src/structure.cc)

## Alignment
SET(SFM_ALIGNMENT_Files src/crossview_alignment.h src/crossview_alignment.cc)

Set(ALL_Files 
    ${Basic_Files}
	${SFM_ALIGNMENT_Files}
	${Utils_HDRS}        ${Utils_SRCS}
	${Alignment_HDRS}    ${Alignment_SRCS})
add_library(CrossViewRegistration STATIC ${ALL_Files})

target_link_libraries(CrossViewRegistration legacy_stdio_definitions.lib shlwapi.lib winmm.lib
${CMAKE_DL_LIBS} ${OpenCV_LIBS} ${CERES_LIBRARIES})
	
add_subdirectory(test)
