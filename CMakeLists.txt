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
)

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

include_directories(. gui ${CMAKE_CURRENT_BINARY_DIR})
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
    ${QT_LIBRARIES} ${OpenCV_LIBS} cvwidgets cvsegmentation_lib
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -std=c++11 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -O3")

add_subdirectory(segmentation)
