#pragma once
// Untruncated categorical sampling. Each lane owns a strided subset of the
// vocabulary; first choose a lane by its mass, then a token within that lane.
// FP64 weights and the same cutoff as the CPU preserve the target distribution.
struct SampleDraw { float temperature; int active; double uniform; };
__device__ __forceinline__ void sample_logits_impl(
    const float* logits, int n, int ld, const SampleDraw* draws, const int* greedy, int* next) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const SampleDraw draw = draws[row];
    if (tid == 0) next[row] = greedy[row];
    if (!draw.active) return;
    const float* lg = logits + (size_t)row * ld;
    const float mx = lg[greedy[row]];
    if (!isfinite(mx)) return; // CPU falls back to argmax for nonfinite maxima.
    const float threshold = __fsub_rn(mx, __fmul_rn(30.f, draw.temperature));
    const double inv_t = 1.0 / (double)draw.temperature;
    double mass = 0.;
    for (int i = tid; i < n; i += 1024) {
        const float l = lg[i];
        if (l >= threshold) mass += exp((double)__fsub_rn(l, mx) * inv_t);
    }
    double prefix = mass;
    for (int d = 1; d < 64; d *= 2) {
        const double other = __shfl_up(prefix, d, 64);
        if (lane >= d) prefix += other;
    }
    __shared__ double totals[16], bases[16], total;
    __shared__ int winner, lasts[16], last_positive;
    const unsigned long long positive = __ballot(mass > 0.);
    if (lane == 63) { totals[wave] = prefix; lasts[wave] = positive ? wave * 64 + 63 - __clzll(positive) : -1; }
    if (tid == 0) winner = 1024;
    __syncthreads();
    if (tid == 0) { last_positive = -1; for (int w = 0; w < 16; w++) last_positive = max(last_positive, lasts[w]); }
    if (wave == 0) {
        double p = lane < 16 ? totals[lane] : 0.;
        for (int d = 1; d < 16; d *= 2) {
            const double other = __shfl_up(p, d, 64);
            if (lane >= d) p += other;
        }
        if (lane < 16) bases[lane] = p - totals[lane];
        if (lane == 15) total = p;
    }
    __syncthreads();
    double r = draw.uniform * total - bases[wave] - (prefix - mass);
    if (mass > 0. && r >= 0. && r < mass) atomicMin(&winner, tid);
    __syncthreads();
    const bool endpoint = winner == 1024 && tid == last_positive;
    if (tid != winner && !endpoint) return;
    if (endpoint) r = nextafter(mass, 0.);
    int last = greedy[row];
    for (int i = tid; i < n; i += 1024) {
        const float l = lg[i];
        if (l < threshold || isnan(l)) continue;
        last = i;
        r -= exp((double)__fsub_rn(l, mx) * inv_t);
        if (r < 0.) { next[row] = i; return; }
    }
    next[row] = last;
}

extern "C" __global__ void __launch_bounds__(1024) sample_logits_b(const float* logits, int n, const SampleDraw* draws, const int* greedy, int* next) {
    sample_logits_impl(logits,n,n,draws,greedy,next);
}
extern "C" __global__ void __launch_bounds__(1024) sample_logits_stride_b(const float* logits, int n, int ld, const SampleDraw* draws, const int* greedy, int* next) {
    sample_logits_impl(logits,n,ld,draws,greedy,next);
}
