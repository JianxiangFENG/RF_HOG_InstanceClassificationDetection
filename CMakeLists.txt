cmake_minimum_required(VERSION 3.5)
PROJECT(TDCV_RF)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   else()
      message(STATUS "OpenCV > 2.4.3 found!!!!!")
   endif()
else()
   message(STATUS "OpenCV > 3.4 found!!!!!")
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

# build and link internal library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
	src/visual_hog.cpp
	src/helper.cpp
	src/BaggingTrees.cpp)

# link external library
target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS})

# build source codes
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(task1 src/task1.cpp)
target_link_libraries(task1 ${PROJECT_NAME})

add_executable(task2 src/task2.cpp)
target_link_libraries(task2 ${PROJECT_NAME})

add_executable(task3 src/task3.cpp)
target_link_libraries(task3 ${PROJECT_NAME})

add_executable(SelectiveSearch src/SelectiveSearch.cpp)
target_link_libraries(SelectiveSearch ${PROJECT_NAME})
