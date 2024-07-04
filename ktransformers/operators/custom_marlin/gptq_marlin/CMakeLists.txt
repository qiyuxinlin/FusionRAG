cmake_minimum_required(VERSION 3.20)
project(MarlinLIB  LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "/opt/conda/lib/python3.10/site-packages/torch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(third_party/pybind11)

set(SOURCES 
  "gptq_marlin/gptq_marlin_repack.cu"
  "gptq_marlin/gptq_marlin.cu"
  "qlib.cpp"
)
pybind11_add_module(qlib MODULE ${SOURCES})
# add_library(MarlinGPTQ SHARED ${SOURCES})
# target_link_libraries(qlib "${TORCH_LIBRARIES}")
target_link_libraries(qlib PRIVATE ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
set_property(TARGET qlib PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)