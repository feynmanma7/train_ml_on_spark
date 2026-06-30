#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;
using namespace torch_lite;

// Helper: Tensor → numpy array
py::array_t<float> tensor_to_numpy(const Tensor& t) {
    std::vector<ssize_t> py_shape(t.shape.begin(), t.shape.end());
    py::array_t<float> arr(py_shape);
    std::memcpy(arr.mutable_data(), t.data.data(), t.numel() * sizeof(float));
    return arr;
}

// Helper: numpy array → Tensor
py::object numpy_to_tensor(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < buf.ndim; i++) {
        shape.push_back((int64_t)buf.shape[i]);
    }
    float* ptr = (float*)buf.ptr;
    std::vector<float> data(ptr, ptr + (int64_t)buf.size);
    return py::cast(new Tensor(data, shape), py::return_value_policy::take_ownership);
}

// Helper: Python nested list → Tensor
Tensor list_to_tensor(const py::list& py_list, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<float> data(n);

    std::function<void(const py::handle&, int64_t&)> flatten;
    flatten = [&](const py::handle& obj, int64_t& idx) {
        if (py::isinstance<py::list>(obj)) {
            for (auto item : py::cast<py::list>(obj)) {
                flatten(item, idx);
            }
        } else {
            data[idx++] = py::cast<float>(obj);
        }
    };
    int64_t idx = 0;
    flatten(py_list, idx);
    return Tensor(data, shape);
}

PYBIND11_MODULE(_torch_lite_core, m) {
    m.doc() = "torch_lite C++ backend - fast tensor operations";

    // === Tensor class ===
    py::class_<Tensor>(m, "Tensor")
        // Constructors (via static factories)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"))
        .def_static("ones", &Tensor::ones, py::arg("shape"))
        .def_static("randn", &Tensor::randn,
            py::arg("shape"), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f, py::arg("seed") = 42)
        .def_static("from_numpy", &numpy_to_tensor, py::arg("arr"))
        .def_static("from_list", &list_to_tensor, py::arg("data"), py::arg("shape"))

        // Properties
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape; })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def("size", &Tensor::size, py::arg("dim"))
        .def("rows", &Tensor::rows)
        .def("cols", &Tensor::cols)

        // Data access
        .def("numpy", &tensor_to_numpy)
        .def("__repr__", &Tensor::str)
        .def("__str__", &Tensor::str)

        // Data pointer for efficient access
        .def_property_readonly("_data", [](const Tensor& t) -> py::bytes {
            return py::bytes(reinterpret_cast<const char*>(t.data_ptr()), t.numel() * sizeof(float));
        })

        // In-place
        .def("fill_", &Tensor::fill_, py::arg("val"))
        .def("zero_", &Tensor::zero_)
        .def("copy_", &Tensor::copy_)

        // Shape
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("t", &Tensor::transpose)
        .def("transpose", &Tensor::transpose);

    // === Free functions ===
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"), "Matrix multiplication");
    m.def("add", &add, py::arg("a"), py::arg("b"), "Element-wise addition");
    m.def("sub", &sub, py::arg("a"), py::arg("b"), "Element-wise subtraction");
    m.def("mul", &mul, py::arg("a"), py::arg("b"), "Element-wise multiplication");
    m.def("elem_div", &elem_div, py::arg("a"), py::arg("b"), "Element-wise division");
    m.def("scale", &scale, py::arg("a"), py::arg("s"), "Scalar multiplication");
    m.def("broadcast_add", &broadcast_add, py::arg("a"), py::arg("bias"),
          "Broadcast add: [M,N] + [1,N] -> [M,N]");

    // Activation
    m.def("relu", &relu, py::arg("x"), "ReLU activation");
    m.def("relu_backward", &relu_backward, py::arg("grad"), py::arg("input"),
          "ReLU backward: grad * (input > 0)");
    m.def("sigmoid", &sigmoid, py::arg("x"), "Sigmoid activation");

    // Reduction
    m.def("sum_dim0", &sum_dim0, py::arg("x"), "Sum along dimension 0: [M,N] -> [1,N]");

    // Loss
    m.def("binary_cross_entropy", &binary_cross_entropy, py::arg("pred"), py::arg("target"),
          "Binary cross entropy loss (pred should be probabilities)");
    m.def("bce_sigmoid_backward", &bce_sigmoid_backward, py::arg("logits"), py::arg("target"),
          "Gradient of BCE(sigmoid(logits), target) w.r.t logits");

    // Utility
    m.def("outer", &outer, py::arg("a"), py::arg("b"), "Outer product [M] * [N] -> [M,N]");
}
