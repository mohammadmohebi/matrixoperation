cmake_minimum_required (VERSION 2.6)

project (tp3)
add_definitions(-std=c++11)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(tp3 main.cpp Matrix.cpp Inverter.cpp)

include(CPack)

if(MPI_COMPILE_FLAGS)
  set_target_properties(tp3 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(tp3 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(tp3 ${MPI_LIBRARIES})