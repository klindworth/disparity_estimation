cmake_minimum_required(VERSION 2.8)

project(disparity_estimation)

find_package(Qt4 REQUIRED QtCore QtGui)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
include(${QT_USE_FILE})


set(SOURCES main.cpp
    genericfunctions.cpp
    slidinghimigradient.cpp
    debugmatstore.cpp
    costmap_utils.cpp
    region.cpp
    initial_disparity.cpp
    stereotask.cpp
    window_size.cpp
    disparity_utils.cpp
    region_optimizer.cpp
    configrun.cpp
    taskanalysis.cpp
    slidingGradient.cpp
    refinement.cpp

    gui/matstoreviewer.cpp
    gui/costmapviewer.cpp
    gui/miniplot.cpp    
    gui/regionwidget.cpp
    gui/imagestore.cpp    
    gui/regionwindow.cpp   

    segmentation/region_descriptor.cpp
    segmentation/segmentation.cpp
    segmentation/segmentation_cr.cpp
)

set(HEADERS
    genericfunctions.h
    fast_array.h
    slidingGradient.h
    slidinghimigradient.h
    debugmatstore.h
    slidingEntropy.h
    costmap_creators.h
    it_metrics.h
    costmap_utils.h
    region.h

    sparse_counter.h
    initial_disparity.h
    stereotask.h

    window_size.h
    disparity_utils.h

    region_optimizer.h
    configrun.h
    taskanalysis.h
    
    misc.h

    region_metrics.h
    refinement.h

    segmentation/region_descriptor.h
    segmentation/region_descriptor_algorithms.h
    segmentation/intervals_algorithms.h
    segmentation/intervals.h
    segmentation/segmentation_algorithms.h
    segmentation/segmentation.h
    segmentation/segmentation_cr.h
    segmentation/segmentation_image.h
)

set(ENABLE_SLIC ON)
set(ENABLE_MEANSHIFT ON)

if(ENABLE_SLIC)
    set(SOURCES ${SOURCES}
        SLIC_CV/slic.cpp
        SLIC_CV/slic_adaptor.cpp
        segmentation/segmentation_slic.cpp
    )

    set(HEADERS ${HEADERS}
        segmentation/segmentation_slic.h
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
        segmentation/segmentation_ms.cpp
    )

    set(HEADERS ${HEADERS}
        meanshift_cv/rlist.h
        meanshift_cv/tdef.h
        meanshift_cv/ms.h
        meanshift_cv/ms_cv.h
        meanshift_cv/msImageProcessor.h
        meanshift_cv/msSys.h
        meanshift_cv/RAList.h
        segmentation/segmentation_ms.h
    )

    add_definitions(-DUSE_MEANSHIFT)
endif()

if(ENABLE_SLIC AND ENABLE_MEANSHIFT)
    set(SOURCES ${SOURCES}
        segmentation/segmentation_msslic.cpp
    )

    set(HEADERS ${HEADERS}
        segmentation/segmentation_msslic.h
    )

    add_definitions(-DUSE_MSSLIC)
endif()

set(QTHEADERS
    gui/matstoreviewer.h
    gui/costmapviewer.h
    gui/miniplot.h
    gui/regionwindow.h
    gui/regionwidget.h
    gui/imagestore.h
)

set(FORMS
    gui/matstoreviewer.ui
    gui/costmapviewer.ui
    gui/regionwidget.ui
    gui/imagestore.ui
    gui/regionwindow.ui
)

qt4_wrap_cpp(HEADERS_MOC ${QTHEADERS})
qt4_wrap_ui(FORMS_HEADERS ${FORMS})
qt4_add_resources(RESOURCES_RCC ${RESOURCES})

include_directories(. gui segmentation ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${QT_DEFINITIONS})

add_executable(${PROJECT_NAME}
    ${QTHEADERS}
    ${HEADERS}
    ${SOURCES}
    ${HEADERS_MOC}
    ${FORMS_HEADERS}
    ${RESOURCES_RCC}
)

target_link_libraries(${PROJECT_NAME}
    ${QT_LIBRARIES} ${OpenCV_LIBS} cvwidgets
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -std=c++11 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -O3")
