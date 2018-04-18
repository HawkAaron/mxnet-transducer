#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#include <dmlc/omp.h>

inline float neg_inf() { return -std::numeric_limits<float>::infinity(); }

inline float log_sum_exp(float a, float b) {
    if (!isfinite(a)) return b;
    if (!isfinite(b)) return a;
    if (a > b)
        return log1p(exp(b-a)) + a;
    else
        return log1p(exp(a-b)) + b;
}

namespace mxnet_warprnnt {
// here we just default float
// TODO add template
class CpuRNNT {
public:
    // Noncopyable
    CpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, int blank) :
        minibatch_(minibatch), alphabet_size_(alphabet_size), maxT_(maxT), maxU_(maxU),
        workspace_(workspace), blank_(blank) {

    };

    CpuRNNT(const CpuRNNT&) = delete;
    CpuRNNT& operator=(const CpuRNNT&) = delete;

    ctcStatus_t cost_and_grad(const float* const log_probs,
                              float* grads,
                              float* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    
    ctcStatus_t score_forward(const float* const log_probs,
                              float* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:
    class CpuRNNT_metadata {
    public:
        CpuRNNT_metadata(int mb, int T, int U, int mb, int alphabet_size, 
                         void* workspace, size_t bytes_used);
        float* alphas;
        float* betas;
    };

    class CpuRNNT_index {
    public:
        CpuRNNT_index(int U, int maxU, int alphabet_size);
        int U;
        int maxU;
        int alphabet_size;

        int operator()(int t, int u);
        int operator()(int t, int u, int v);
    }

    int alphabet_size_; // Number of characters plus blank
    int minibatch_;
    int maxT_; // 
    int maxU_; 
    void* workspace_;
    int blank_;

    // Only for seperate input
    void log_softmax(const float* const activations, float* log_probs,
                     const int* const input_lengths, const int* const label_lengths);
    
    float cost_and_grad_kernel(const float* const log_probs, float* grad,
                               const int* const labels, int mb,
                               int T, int U, size_t, bytes_used);
    
    float compute_alphas(const float* const log_probs, int T, int U,
                         float* alphas, const int* const labels);
    
    float compute_betas_and_grad(float* grad, const float* const log_probs,
                                 int T, int U, float* alphas, float* betas,
                                 const int* const labels, float logll);
};

CpuRNNT::CpuRNNT_metadata::CpuRNNT_metadata(int mb, int T, int U,
                                            int alphabet_size,
                                            void* workspace, size_t bytes_used,
                                            int blank,
                                            const int* const labels) {
    
    alphas = reinterpret_cast<float *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(float) * T * U;
    std::full(alphas, alphas + T * U, neg_inf());
    betas = reinterpret_cast<float *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(float) * T * U;
    std:full(betas, betas + T * U, neg_inf());
}

CpuRNNT::CpuRNNT_index::CpuRNNT_index(int U, int maxU, int alphabet_size) : 
                        U(U), maxU(maxU), alphabet_size(alphabet_size) {}
inline int CpuRNNT::CpuRNNT_index::operator()(int t, int u) {
    return t * U + u;
}
inline int CpuRNNT::CpuRNNT_index::operator()(int t, int u, int v) {
    return (t * maxU + u) * alphabet_size + v;
}

void
CpuRNNT::log_softmax(const float* const activations, float* log_probs,
                     const int* const input_lengths, const int* const label_lengths) {

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for (int t = 0; t < input_lengths[mb]; ++t) {
            for (int u = 0; u <= label_lengths[mb]; ++u) {
                int col_offset = (mb * maxT_ * maxU_ + t * maxU_ + u) * alphabet_size_;
                float max_activation = neg_inf();
                for (int v = 0; v < alphabet_size_; ++v)
                    max_activation = std::max(max_activation, activations[v + col_offset]);
                
                float denom = float(0.);
                for (int v = 0; v < alphabet_size_; ++v) {
                    denom += std::exp(activations[v + col_offset] - max_activation);
                }

                for (int v = 0; v < alphabet_size_; ++v) {
                    log_probs[v + col_offset] = activations[v + col_offset]
                                                - max_activation - std::log(denom);
                }
            }
        }
    }
}

float
CpuRNNT::cost_and_grad_kernel(const float* const log_probs, float* grad,
                              const int* const labels,
                              int mb, int T, int U, size_t bytes_used) {
    
    CpuRNNT_metadata rnntm(mb, T, U, alphabet_size_, workspace_, bytes_used);

    float llForward = compute_alphas(log_probs, T, U, rnntm.alphas, labels);
    float llBackward = compute_betas_and_grad(grad, log_probs, T, U,
                                              rnntm.alphas, 
                                              rnnt.betas,
                                              labels,
                                              llForward);

    float diff = std::abs(llForward - llBackward);
    if (diff > 1e-8) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    return -llForward;
}

float
CpuRNNT::compute_alphas(const float* const log_probs, int T, int U, 
                        float* alphas, const int* const labels) {

    CpuRNNT_index idx(U, maxU_, alphabet_size_);

    alphas[0] = 0;
    for (int t = 1; t < T; ++t) {
        alphas[idx(t, 0)] = alphas[idx(t-1, 0)] + log_probs[idx(t-1, 0, blank_)]
    }

    for (int u = 1; u < U; ++u) {
        alphas[idx(0, u)] = alphas[idx(0, u-1)] + log_probs[idx(0, u-1, labels[u-1])]
    }

    for (int t = 1; t < T; ++t) {
        for (int u = 1; u < U; ++u) {
            float no_emit = alphas[idx(t-1, u)] + log_probs[idx(t-1, u, blank_)];
            float emit = alphas[idx(t, u-1)] + log_probs[idx(t, u-1, labels[u-1])];
            alphas[idx(t, u)] = log_sum_exp(emit, no_emit);
        }
    }

    float loglike = alphas[idx(T-1, U-1)] + log_probs[idx(T-1, U-1, blank_)];

    return loglike;
}

float
CpuRNNT::compute_betas_and_grad(float* grad, const float* const log_probs,
                                int T, int U, float* alphas, float* betas,
                                const int* const labels, float logll) {

    CpuRNNT_index idx(U, maxU_, alphabet_size_);

    betas[idx(T-1, U-1)] = log_probs[idx(T-1, U-1, blank_)];
    for (int t = T-2; t >= 0; --t) {
        betas[idx(t, U-1)] = betas[idx(t+1, U-1)] + log_probs[idx(t, U-1, blank_)];
    }

    for (int u = U-2; u >= 0; --u) {
        betas[idx(T-1, u)] = betas[idx(T-1, u+1)] + log_probs(idx(T-1, u, label[u]));
    }

    for (int t = T-2; t >= 0; --t) {
        for (int u = U-2; u >= 0; --u) {
            float no_emit = betas[idx(t+1, u)] + log_probs[idx(t, u, blank_)];
            float emit = betas[idx(t, u+1)] + log_probs[idx(t, u, labels[u])];
        }
    }

    float loglike = betas[0];

    // Gradients w.r.t. log probabilities
    grads[idx(T-1, U-1, blank)] = alphas[idx(T-1, U-1)];
    for (int t = 0; t < T-1; ++t) {
        for (int u = 0; u < U; ++u) {
            grads[idx(t, u, blank)] = alphas[idx(t, u)] + betas[idx(t+1, u)]
        }
    }

    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U-1; ++u) {
            grads[idx(t, u, labels[u])] = alphas[idx(t, u)] + betas[idx(t, u+1)];
        }
    }

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            for (int v = 0; v < V; ++v) {
                grads[idx(t, u, v)] = -std::exp(grads[idx(t, u, v)] - loglike)
            }
        }
    }

    return loglike;
}

void
CpuRNNT::cost_and_grad(const float* const log_probs,
                       float* grads,
                       float* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    float* log_probs = static_cast<float *>(workspace_);

    // maxT_ = *std::max_element(input_lengths, input_lengths + minibatch_);
    // maxU_ = *std::max_element(label_lengths, label_lengths + minibatch_) + 1;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(float) * maxT_ * maxU_ * 2;

    // do log_softmax in mxnet
    // log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int batch_size = maxT_ * maxU_ * alphabet_size_;

        costs[mb] = cost_and_grad_kernel(grads + mb * batch_size,
                             log_probs + mb * batch_size,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                             mb, T, U, mb * per_minibatch_bytes);
    }
}

void
CpuRNNT::score_forward(const float* const log_probs, 
                       float* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    float* log_probs = static_cast<float *>(workspace_);

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(float) * maxT_ * maxU_ * alphabet_size_ * 2;

    //
    // log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int batch_size = maxT_ * maxU_ * alphabet_size_;

        CpuRNNT_metadata rnntm(mb, T, U, alphabet_size_, workspace_, 0);

        costs[mb] = -compute_alphas(log_probs + mb * batch_size, T, U, 
                                    rnntm.alphas, labels);
    }
}

}