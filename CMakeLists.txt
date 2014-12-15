cmake_minimum_required(VERSION 2.8)

project(disparity_estimation)

find_package(Qt4 REQUIRED QtCore QtGui)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem system)
find_package(OpenMP)
include(${QT_USE_FILE})


set(SOURCES main.cpp
    genericfunctions.cpp
    slidinghimigradient.cpp
    debugmatstore.cpp
    costmap_utils.cpp
    disparity_region.cpp
    initial_disparity.cpp
    stereotask.cpp
    window_size.cpp
    disparity_utils.cpp
    region_optimizer.cpp
    configrun.cpp
    taskanalysis.cpp
    slidingGradient.cpp
    refinement.cpp
    sncc_disparitywise_calculator.cpp
    manual_region_optimizer.cpp
    ml_region_optimizer.cpp

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
    disparity_region.h
    disparity_region_algorithms.h
    disparitywise_calculator.h
    sncc_disparitywise_calculator.h

    sparse_counter.h
    initial_disparity.h
    stereotask.h

    window_size.h
    disparity_utils.h

    region_optimizer.h
    manual_region_optimizer.h
    ml_region_optimizer.h
    configrun.h
    taskanalysis.h
    
    misc.h

    region_metrics.h
    refinement.h

    simple_nn.h
)

set(QTHEADERS
    gui/miniplot.h
    gui/regionwindow.h
    gui/regionwidget.h
    gui/imagestore.h
)

set(FORMS
    gui/regionwidget.ui
    gui/imagestore.ui
    gui/regionwindow.ui
)

qt4_wrap_cpp(HEADERS_MOC ${QTHEADERS})
qt4_wrap_ui(FORMS_HEADERS ${FORMS})
qt4_add_resources(RESOURCES_RCC ${RESOURCES})

include_directories(. gui loganalyzer ${CMAKE_CURRENT_BINARY_DIR})
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
    ${QT_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} cvwidgets cvsegmentation_lib openblas
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -O0 -fsanitize=address -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -O3")

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}_tests simple_nn_tests.cpp simple_nn.h)
target_link_libraries(${PROJECT_NAME}_tests ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} openblas)

add_subdirectory(segmentation)

#analyzer

set(SOURCES_ANALYZER
    loganalyzer/main.cpp
    loganalyzer/analyzer.cpp
    loganalyzer/comparewidget.cpp
    loganalyzer/detailviewer.cpp

    genericfunctions.cpp
    disparity_utils.cpp
    taskanalysis.cpp
    stereotask.cpp
)

set(QTHEADERS_ANALYZER
    loganalyzer/analyzer.h
    loganalyzer/comparewidget.h
    loganalyzer/detailviewer.h
)

set(FORMS_ANALYZER
    loganalyzer/analyzer.ui
    loganalyzer/detailviewer.ui
)

qt4_wrap_cpp(HEADERS_ANALYZER_MOC ${QTHEADERS_ANALYZER})
qt4_wrap_ui(FORMS_ANALYZER_HEADERS ${FORMS_ANALYZER})
#qt4_add_resources(RESOURCES_RCC ${RESOURCES})

add_executable(${PROJECT_NAME}_loganalyzer ${SOURCES_ANALYZER} ${HEADERS_ANALYZER_MOC} ${FORMS_ANALYZER_HEADERS})

target_link_libraries(${PROJECT_NAME}_loganalyzer
    ${QT_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} cvwidgets cvsegmentation_lib
)
