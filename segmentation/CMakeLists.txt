cmake_minimum_required(VERSION 2.8)

project(cvsegmentation_lib)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

set(SOURCES
    region_descriptor.cpp
    segmentation.cpp
    segmentation_cr.cpp
    segmentation_refinement.cpp
)

set(HEADERS
    region_descriptor.h
    region_descriptor_algorithms.h
    intervals_algorithms.h
    intervals.h
    segmentation_algorithms.h
    segmentation.h
    segmentation_cr.h
    segmentation_image.h
    segmentation_refinement.h
)

option(ENABLE_SLIC "Enables SLIC. Forces you to use the GPL as license" ON)
option(ENABLE_MEANSHIFT "Enables the original meanshift implementation of the EDISON library" ON)

if(ENABLE_SLIC)
    set(SOURCES ${SOURCES}
        SLIC_CV/slic.cpp
        SLIC_CV/slic_adaptor.cpp
        segmentation_slic.cpp
    )

    set(HEADERS ${HEADERS}
        segmentation_slic.h
        SLIC_CV/slic.h
        SLIC_CV/slic_adaptor.h
    )

    add_definitions(-DUSE_SLIC)
    message("Remember: With the use of SLIC, this software is licensed under the conditions of the GPL")
endif()

add_subdirectory(meanshift_cv)

if(ENABLE_MEANSHIFT)
    set(SOURCES ${SOURCES}
        meanshift_cv/rlist.cpp
        meanshift_cv/ms.cpp
        meanshift_cv/ms_cv.cpp
        meanshift_cv/msImageProcessor.cpp
        meanshift_cv/msSysPrompt.cpp
        meanshift_cv/RAList.cpp
        segmentation_ms.cpp
    )

    set(HEADERS ${HEADERS}
        meanshift_cv/rlist.h
        meanshift_cv/tdef.h
        meanshift_cv/ms.h
        meanshift_cv/ms_cv.h
        meanshift_cv/msImageProcessor.h
        meanshift_cv/msSys.h
        meanshift_cv/RAList.h
        segmentation_ms.h
    )

    add_definitions(-DUSE_MEANSHIFT)
endif()

if(ENABLE_SLIC AND ENABLE_MEANSHIFT)
    set(SOURCES ${SOURCES}
        segmentation_msslic.cpp
    )

    set(HEADERS ${HEADERS}
        segmentation_msslic.h
    )

    add_definitions(-DUSE_MSSLIC)
endif()

add_library(${PROJECT_NAME} SHARED
    ${HEADERS}
    ${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -std=c++11 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -O3")

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}_tests tests.cpp ${SOURCES})
target_link_libraries(${PROJECT_NAME}_tests ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})

