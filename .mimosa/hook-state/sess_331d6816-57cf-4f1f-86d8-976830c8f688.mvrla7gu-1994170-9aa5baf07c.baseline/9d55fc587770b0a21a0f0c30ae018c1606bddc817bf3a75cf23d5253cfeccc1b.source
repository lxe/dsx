"""patch_qtf_opt.py DST: install the dense fractional-K quantizer (qt_frac_opt.cuh, next to this script) into the
exllamav3 tree DST (v1.6.4, after make_mul16_exllamav3.sh's codebook patch). quantize_tiles_frac then launches
quantize_tiles_frac_opt_kernel; EXL3_QTF_OPT=0 selects exllamav3's generic kernel again (bit-identical indices)."""
import os, shutil, sys

dst = sys.argv[1]
q = os.path.join(dst, "exllamav3", "exllamav3_ext", "quant")
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qt_frac_opt.cuh"), os.path.join(q, "qt_frac_opt.cuh"))


def edit(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))


inst = os.path.join(q, "comp_units", "quantize_tiles_frac_inst.cu")
edit(inst, '#include "../quantize_tiles_frac_kernel.cuh"\n',
     '#include "../quantize_tiles_frac_kernel.cuh"\n#include "../qt_frac_opt.cuh"\n')
s = open(inst).read()
s += "".join(f"fp_quantize_tiles_kernel quantize_tiles_frac_opt_kernel_a{a}_maaaa() "
             f"{{ return quantize_tiles_frac_opt_kernel<{a}, 0xaaaau, 1024>; }}\n" for a in (1, 2, 3))
open(inst, "w").write(s)

edit(os.path.join(q, "comp_units", "quantize_tiles_instances.cuh"),
     "fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa();\n",
     "fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa();\n"
     "// Dense fractional-rate instances (qt_frac_opt.cuh, T16 patch): 1024 threads, costs in shared memory\n"
     "fp_quantize_tiles_kernel quantize_tiles_frac_opt_kernel_a1_maaaa();\n"
     "fp_quantize_tiles_kernel quantize_tiles_frac_opt_kernel_a2_maaaa();\n"
     "fp_quantize_tiles_kernel quantize_tiles_frac_opt_kernel_a3_maaaa();\n")

cu = os.path.join(q, "quantize.cu")
edit(cu, """    fp_quantize_tiles_kernel kernel = nullptr;
    struct { int ka; uint32_t mask; fp_quantize_tiles_kernel (*fn)(); } const table[] = {
        {1, 0xaaaau, &quantize_tiles_frac_kernel_a1_maaaa},
        {2, 0xaaaau, &quantize_tiles_frac_kernel_a2_maaaa},
        {3, 0xaaaau, &quantize_tiles_frac_kernel_a3_maaaa},
    };
    for (const auto& e : table) if (e.ka == KA && e.mask == (uint32_t) MASK) kernel = e.fn();""",
     """    // T16 patch: the dense kernel (qt_frac_opt.cuh) unless EXL3_QTF_OPT=0; both give the same indices
    const char* opt_env = std::getenv("EXL3_QTF_OPT");
    const bool opt = !(opt_env && opt_env[0] == '0');
    fp_quantize_tiles_kernel kernel = nullptr;
    struct { int ka; uint32_t mask; fp_quantize_tiles_kernel (*fn)(); fp_quantize_tiles_kernel (*fn_opt)(); } const table[] = {
        {1, 0xaaaau, &quantize_tiles_frac_kernel_a1_maaaa, &quantize_tiles_frac_opt_kernel_a1_maaaa},
        {2, 0xaaaau, &quantize_tiles_frac_kernel_a2_maaaa, &quantize_tiles_frac_opt_kernel_a2_maaaa},
        {3, 0xaaaau, &quantize_tiles_frac_kernel_a3_maaaa, &quantize_tiles_frac_opt_kernel_a3_maaaa},
    };
    for (const auto& e : table) if (e.ka == KA && e.mask == (uint32_t) MASK) kernel = opt ? e.fn_opt() : e.fn();""")
edit(cu, """    const int shmem = L * sizeof(half) + 32 * sizeof(int) + 128;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    const int max_batch_size = (int) MIN(temp_costs.size(0), temp_edges.size(0));""",
     """    // The dense kernel: shared memory = input tile, argmin scratch, two cost buffers (65536 >> KA and
    // 65536 >> (KA + 1) halves); its traceback history (1 MB per tile) reuses temp_edges, temp_costs is unused
    const int threads = opt ? 1024 : 512;
    const int shmem = opt ? L * (int) sizeof(half) + 32 * (int) sizeof(int) + (65536 >> KA) * 2 + (65536 >> (KA + 1)) * 2
                          : L * sizeof(half) + 32 * sizeof(int) + 128;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    const int max_batch_size = opt ? (int) MIN(temp_edges.numel() * 2 / (256 * 1024 * 4), (int64_t) 65535)
                                   : (int) MIN(temp_costs.size(0), temp_edges.size(0));
    TORCH_CHECK(max_batch_size > 0, "quantize_tiles_frac: scratch must hold at least one tile");""")
edit(cu, "        kernel<<<bsz, 512, shmem, stream>>>", "        kernel<<<bsz, threads, shmem, stream>>>")
print("qt_frac_opt installed in", dst)
