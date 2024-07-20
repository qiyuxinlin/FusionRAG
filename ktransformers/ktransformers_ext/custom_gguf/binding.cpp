#include "ops.h"
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;
// using namespace qlib;

int test(){
    return 5;
}


PYBIND11_MODULE(cudaops, m) {
    m.def("dequantize_q8_0", &dequantize_q8_0, "Function to dequantize q8_0 data.",
          py::arg("data"), py::arg("blk_size"), py::arg("device"));
    // m.def("dequantize_q4_k", &dequantize_q4_k, "Function to dequantize q4_k data.",
    //       py::arg("data"));
    // m.def("dequantize_q6_k", &dequantize_q6_k, "Function to dequantize q6_k data.",
    //       py::arg("data"));
    m.def("test", &test, "Function to test.");
    m.def("compare", &compare,
        py::arg("data"), py::arg("blk_size"), py::arg("device"), py::arg("cmp_scales"), py::arg("cmp_qs"));
}
