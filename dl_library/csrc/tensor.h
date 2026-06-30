#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <random>
#include <memory>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace torch_lite {

class Tensor {
public:
    std::vector<float> data;
    std::vector<int64_t> shape;  // [rows, cols]

    Tensor() = default;

    // Create from shape (uninitialized)
    Tensor(const std::vector<int64_t>& shape_)
        : shape(shape_) {
        int64_t n = numel();
        data.resize(n, 0.0f);
    }

    // Create from data + shape
    Tensor(const std::vector<float>& data_, const std::vector<int64_t>& shape_)
        : data(data_), shape(shape_) {}

    // --- Accessors ---
    inline float& at(int64_t i, int64_t j) {
        return data[i * shape[1] + j];
    }
    inline float at(int64_t i, int64_t j) const {
        return data[i * shape[1] + j];
    }

    inline int64_t rows() const { return shape.size() > 0 ? shape[0] : 1; }
    inline int64_t cols() const { return shape.size() > 1 ? shape[1] : 1; }
    inline int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    inline int64_t ndim() const { return (int64_t)shape.size(); }
    inline int64_t size(int dim) const { return shape[dim]; }

    std::string str() const {
        std::ostringstream oss;
        oss << "Tensor([";
        for (int64_t i = 0; i < rows(); i++) {
            if (i > 0) oss << ",\n       ";
            oss << "[";
            for (int64_t j = 0; j < cols(); j++) {
                if (j > 0) oss << ", ";
                float v = at(i, j);
                if (std::abs(v) < 1e-6f) v = 0.0f;  // clean near-zero
                oss << v;
            }
            oss << "]";
        }
        oss << "], shape=[" << rows() << ", " << cols() << "])";
        return oss.str();
    }

    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }

    // --- Factory methods ---
    static Tensor zeros(const std::vector<int64_t>& shape);
    static Tensor ones(const std::vector<int64_t>& shape);
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float stddev = 1.0f, unsigned seed = 42);
    static Tensor from_list(const std::vector<float>& data, const std::vector<int64_t>& shape);

    // --- In-place operations ---
    void fill_(float val);
    void zero_();
    void copy_(const Tensor& other);

    // --- Shape operations ---
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor transpose() const;  // [M,N] -> [N,M]
};

// --- Math operations (return new Tensor) ---
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);       // element-wise
Tensor elem_div(const Tensor& a, const Tensor& b);  // element-wise
Tensor scale(const Tensor& a, float s);             // scalar multiply
Tensor broadcast_add(const Tensor& a, const Tensor& bias);  // [M,N] + [N] -> [M,N]

// --- Activation functions ---
Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& grad, const Tensor& input);  // grad * (input > 0)
Tensor sigmoid(const Tensor& x);

// --- Reduction ---
Tensor sum_dim0(const Tensor& x);  // [M,N] -> [N]

// --- Loss ---
float binary_cross_entropy(const Tensor& pred, const Tensor& target);
Tensor bce_sigmoid_backward(const Tensor& pred, const Tensor& target);  // (sigmoid(pred) - target) / N

// --- Utility ---
Tensor outer(const Tensor& a, const Tensor& b);  // [M] * [N] -> [M,N]

}  // namespace torch_lite
