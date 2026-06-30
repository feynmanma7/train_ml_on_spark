#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace torch_lite {

// ===== Factory =====
Tensor Tensor::zeros(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    std::fill(t.data.begin(), t.data.end(), 0.0f);
    return t;
}
Tensor Tensor::ones(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    std::fill(t.data.begin(), t.data.end(), 1.0f);
    return t;
}
Tensor Tensor::randn(const std::vector<int64_t>& shape, float mean, float stddev, unsigned seed) {
    Tensor t(shape);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& v : t.data) v = dist(gen);
    return t;
}
Tensor Tensor::from_list(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Tensor(data, shape);
}

// ===== In-place =====
void Tensor::fill_(float val) {
    std::fill(data.begin(), data.end(), val);
}
void Tensor::zero_() { fill_(0.0f); }
void Tensor::copy_(const Tensor& other) {
    data = other.data;
    shape = other.shape;
}

// ===== Shape =====
Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    return Tensor(data, new_shape);
}
Tensor Tensor::transpose() const {
    int64_t r = rows(), c = cols();
    std::vector<float> out_data(r * c);
    for (int64_t i = 0; i < r; i++)
        for (int64_t j = 0; j < c; j++)
            out_data[j * r + i] = at(i, j);
    return Tensor(out_data, {c, r});
}

// ===== matmul =====
Tensor matmul(const Tensor& a, const Tensor& b) {
    int64_t M = a.rows(), K = a.cols();
    int64_t K2 = b.rows(), N = b.cols();
    if (K != K2) {
        throw std::runtime_error("matmul shape mismatch: (" + std::to_string(M) + "," +
            std::to_string(K) + ") @ (" + std::to_string(K2) + "," + std::to_string(N) + ")");
    }
    Tensor out = Tensor::zeros({M, N});
    for (int64_t i = 0; i < M; i++) {
        for (int64_t k = 0; k < K; k++) {
            float aik = a.at(i, k);
            if (aik == 0.0f) continue;
            for (int64_t j = 0; j < N; j++) {
                out.at(i, j) += aik * b.at(k, j);
            }
        }
    }
    return out;
}

// ===== Element-wise ops =====
Tensor add(const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    Tensor out(a.shape);
    for (int64_t i = 0; i < n; i++) out.data[i] = a.data[i] + b.data[i];
    return out;
}
Tensor sub(const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    Tensor out(a.shape);
    for (int64_t i = 0; i < n; i++) out.data[i] = a.data[i] - b.data[i];
    return out;
}
Tensor mul(const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    Tensor out(a.shape);
    for (int64_t i = 0; i < n; i++) out.data[i] = a.data[i] * b.data[i];
    return out;
}
Tensor elem_div(const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    Tensor out(a.shape);
    for (int64_t i = 0; i < n; i++) out.data[i] = a.data[i] / b.data[i];
    return out;
}
Tensor scale(const Tensor& a, float s) {
    int64_t n = a.numel();
    Tensor out(a.shape);
    for (int64_t i = 0; i < n; i++) out.data[i] = a.data[i] * s;
    return out;
}

// ===== broadcast_add: [M,N] + [N] -> [M,N] =====
Tensor broadcast_add(const Tensor& a, const Tensor& bias) {
    int64_t M = a.rows(), N = a.cols();
    Tensor out(a.shape);
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            out.at(i, j) = a.at(i, j) + bias.at(0, j);
    return out;
}

// ===== Activation =====
Tensor relu(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); i++)
        out.data[i] = x.data[i] > 0.0f ? x.data[i] : 0.0f;
    return out;
}
Tensor relu_backward(const Tensor& grad, const Tensor& input) {
    Tensor out(grad.shape);
    for (size_t i = 0; i < grad.data.size(); i++)
        out.data[i] = input.data[i] > 0.0f ? grad.data[i] : 0.0f;
    return out;
}
Tensor sigmoid(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); i++) {
        float v = x.data[i];
        // numerically stable: clamp to avoid overflow
        if (v >= 20.0f) out.data[i] = 1.0f;
        else if (v <= -20.0f) out.data[i] = 0.0f;
        else out.data[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return out;
}

// ===== Reduction =====
Tensor sum_dim0(const Tensor& x) {
    int64_t M = x.rows(), N = x.cols();
    Tensor out = Tensor::zeros({1, N});
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            out.at(0, j) += x.at(i, j);
    return out;
}

// ===== Loss =====
float binary_cross_entropy(const Tensor& pred, const Tensor& target) {
    float loss = 0.0f;
    int64_t N = pred.rows();
    for (int64_t i = 0; i < N; i++) {
        float p = pred.at(i, 0);
        float y = target.at(i, 0);
        // clamp for numerical stability
        p = std::max(std::min(p, 1.0f - 1e-7f), 1e-7f);
        loss += -y * std::log(p) - (1.0f - y) * std::log(1.0f - p);
    }
    return loss / (float)N;
}

Tensor bce_sigmoid_backward(const Tensor& logits, const Tensor& target) {
    // d(BCE(sigmoid(logits), target)) / d(logits) = (sigmoid(logits) - target) / N
    Tensor out(logits.shape);
    int64_t N = logits.rows();
    for (int64_t i = 0; i < N; i++) {
        float x = logits.at(i, 0);
        float s;
        if (x >= 20.0f) s = 1.0f;
        else if (x <= -20.0f) s = 0.0f;
        else s = 1.0f / (1.0f + std::exp(-x));
        out.at(i, 0) = (s - target.at(i, 0)) / (float)N;
    }
    return out;
}

// ===== Outer product: [M] * [N] -> [M,N] =====
Tensor outer(const Tensor& a, const Tensor& b) {
    int64_t M = a.numel(), N = b.numel();
    Tensor out = Tensor::zeros({M, N});
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            out.at(i, j) = a.data[i] * b.data[j];
    return out;
}

}  // namespace torch_lite
