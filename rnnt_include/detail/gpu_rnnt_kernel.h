#pragma once

#include "rnnt_helper.h"

namespace warp_rnnt {

template<typename T>
inline __device__ T logp(const T* const denom, const T* const acts, const int maxT, const int maxU, const int alphabet_size, int mb, int t, int u, int v) {
    const int col = (mb * maxT + t) * maxU + u;
    return denom[col] + acts[col * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* nllForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); // mb label start point
    const int offset = tid * maxT * maxU;
    alphas += offset;
    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, 0, blank_);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1]);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);
    nllForward[tid] = -loglike;
    __syncthreads();
}

template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* nllBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, U-1, blank_);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, u, labels[u]);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    nllBackward[tid] = -betas[0];
    __syncthreads();
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp* const acts, const Tp* const denom, const Tp* const alphas, const Tp* const betas, const Tp* const nll, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
            Tp grad = exp(alphas[col] + betas[col] + logpk + nll[mb]);
            // grad to last blank transition
            if (idx == blank_ && t == T-1 && u == U-1) grad -= 1;
            if (idx == blank_ && t < T-1) {
                grad -= exp(alphas[col] + logpk + nll[mb] + betas[col + maxU]);
            }
            if (idx == labels[u] && u < U-1) {
                grad -= exp(alphas[col] + logpk + nll[mb] + betas[col+1]);
            }
            grads[col * alphabet_size + idx] = grad;

            idx += NT;
        }
    }
    __syncthreads();
}

} // warp_rnnt
