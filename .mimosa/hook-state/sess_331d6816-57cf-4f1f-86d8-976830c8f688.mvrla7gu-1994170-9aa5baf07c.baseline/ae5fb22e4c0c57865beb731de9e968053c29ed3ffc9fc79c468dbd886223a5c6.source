// Lucebox 1017 eval for llama.cpp models, the same metric as dsx's `eval`: 8 x 1024 held-out tokens teacher-forced,
// per position KL(teacher || model) over the FP8 teacher's top-256 log-probs, top-1 agreement, perplexity.
// Usage: lucebox-eval <llama.cpp common args: -m MODEL, -ngl, --load-mode, ...> ; env LUCEBOX_EVAL=DIR (heldout.i32,
// teacher_idx.i32, teacher_lp.f32 from dsx's tools/eval_bins.py), LUCEBOX_SEQS (default 8), LUCEBOX_FROM (default 512:
// a second line scores only predictions of positions >= FROM, like `dsx eval --from`).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

template <class T> static std::vector<T> rd(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    size_t n = f.tellg();
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read((char *) v.data(), n);
    return v;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 1024;
    params.n_batch = 1024;
    params.n_ubatch = 1024;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) return 1;
    const char * dir = getenv("LUCEBOX_EVAL");
    const int seqs = getenv("LUCEBOX_SEQS") ? atoi(getenv("LUCEBOX_SEQS")) : 8;
    const int from = getenv("LUCEBOX_FROM") ? atoi(getenv("LUCEBOX_FROM")) : 512;
    const int len = 1024, topk = 256;
    auto toks = rd<int32_t>(std::string(dir) + "/heldout.i32");
    auto tidx = rd<int32_t>(std::string(dir) + "/teacher_idx.i32");
    auto tlp = rd<float>(std::string(dir) + "/teacher_lp.f32");
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "load failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    double nll = 0, kl = 0;
    long agree = 0, cnt = 0;
    double nll2 = 0, kl2 = 0; // predictions of positions >= from
    long agree2 = 0, cnt2 = 0;
    std::vector<double> lq(nv);
    for (int s = 0; s < seqs; s++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(len, 0, 1);
        for (int p = 0; p < len; p++) {
            b.token[p] = toks[s * len + p];
            b.pos[p] = p;
            b.n_seq_id[p] = 1;
            b.seq_id[p][0] = 0;
            b.logits[p] = 1;
        }
        b.n_tokens = len;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int p = 0; p < len - 1; p++) {
            const float * l = llama_get_logits_ith(ctx, p);
            double mx = -INFINITY;
            int am = 0;
            for (int i = 0; i < nv; i++) if (l[i] > mx) { mx = l[i]; am = i; }
            double se = 0;
            for (int i = 0; i < nv; i++) se += std::exp((double) l[i] - mx);
            const double lse = mx + std::log(se);
            const double nl = -(l[toks[s * len + p + 1]] - lse);
            const size_t row = (size_t) s * (len - 1) + p;
            double k = 0;
            for (int j = 0; j < topk; j++) {
                const double lt = tlp[row * topk + j];
                k += std::exp(lt) * (lt - (l[tidx[row * topk + j]] - lse));
            }
            nll += nl;
            kl += k;
            agree += am == tidx[row * topk];
            cnt++;
            if (p + 1 >= from) {
                nll2 += nl;
                kl2 += k;
                agree2 += am == tidx[row * topk];
                cnt2++;
            }
        }
        llama_batch_free(b);
        fprintf(stderr, "seq %d: running ppl %.3f kl %.4f top1 %.4f\n", s, std::exp(nll / cnt), kl / cnt, (double) agree / cnt);
    }
    printf("{\"tokens\": %ld, \"nll\": %.6f, \"ppl\": %.4f, \"kl_top256\": %.6f, \"top1_agree\": %.6f}\n", cnt, nll / cnt, std::exp(nll / cnt), kl / cnt, (double) agree / cnt);
    if (cnt2) printf("{\"from\": %d, \"tokens\": %ld, \"nll\": %.6f, \"ppl\": %.4f, \"kl_top256\": %.6f, \"top1_agree\": %.6f}\n", from, cnt2, nll2 / cnt2, std::exp(nll2 / cnt2), kl2 / cnt2, (double) agree2 / cnt2);
    return 0;
}
