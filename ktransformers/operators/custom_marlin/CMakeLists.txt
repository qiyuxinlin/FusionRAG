cmake_minimum_required(VERSION 3.20)
project(DeepseekV2-local VERSION 0.1.0)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED on)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

add_compile_options(-Wall -DGLOG_USE_GLOG_EXPORT)

# add third-party dependencies
set(WITH_GFLAGS OFF CACHE BOOL "Enable GFlags" FORCE)

add_subdirectory(third_party/glog)

add_subdirectory(third_party/googletest)

add_subdirectory(third_party/benchmark)

# add_subdirectory(third_party/pcinfer)

add_subdirectory(gptq_marlin)

include_directories(SYSTEM third_party/benchmark/include)

set(PYTHON_EXECUTABLE "python3")
add_subdirectory(third_party/pybind11)

# set source

# find OpenMP
find_package(OpenMP REQUIRED)

# find libtorch
# execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE Torch_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
# message(STATUS "Torch_CMAKE_PREFIX_PATH: ${Torch_CMAKE_PREFIX_PATH}")
# find_package(Torch REQUIRED PATHS ${Torch_CMAKE_PREFIX_PATH})
# include_directories(${TORCH_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(qlib MODULE qlib.cpp)
target_link_libraries(qlib PRIVATE OpenMP::OpenMP_CXX MarlinGPTQ)
target_compile_options(qlib PRIVATE -O3 -mtune=native)
target_compile_options(qlib PRIVATE -march=native)
# target_compile_options(qlib PRIVATE -mavx2 -mfma)