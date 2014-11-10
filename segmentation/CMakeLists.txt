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

set(ENABLE_SLIC ON)
set(ENABLE_MEANSHIFT ON)

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
    ${OpenCV_LIBS} cvwidgets
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -std=c++11 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -O3")
