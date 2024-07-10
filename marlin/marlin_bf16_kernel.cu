#ifndef MARLIN_BF16_KERNEL_CUH
#define MARLIN_BF16_KERNEL_CUH

#include <cuda.h>
// #include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

#endif

constexpr int ceildiv(int a, int b)
{
    return (a + b - 1) / b;
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T &operator[](int i)
    {
        return elems[i];
    }
};

// Matrix fragments for tensor core instructions; their precise layout is documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<__nv_bfloat162, 4>;
using FragB = Vec<__nv_bfloat162, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<__nv_bfloat162, 1>;

// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr, bool pred = true)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy with a cache hint indicating that the values may be evicted immediately; used for
// quantized weights B, which are only accessed precisely once and should thus not pollute the L2 cache which we need
// for inputs A and outputs C.
__device__ inline void cp_async4_stream(void *smem_ptr, const void *glob_ptr)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core mma instruction with bf16 inputs and fp32 output/accumulation.
__device__ inline void mma(const FragA &a_frag, const FragB &frag_b, FragC &frag_c)
{
    const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
    const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
    float *c = reinterpret_cast<float *>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA &frag_a, const void *smem_ptr)
{
    uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c)
{
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q)
{
    /*
    The dequant in marlin_cuda_kernel
    FP16:       _  _ _ _ _ _  _ _ _ _ _ _ _ _ _ _
    0x6400      _  1 1 0 0 1  0 0 _ _ _ _ _ _ _ _
                   25
                   exp = 25 - 15 = 10
    q, LO, EX   0  1 1 0 0 1  0 0 0 0 0 0 ? ? ? ?
                   2^10 * (1. 0 + ???? / 2^10) = 2^10 + ????
    0x6408      0  1 1 0 0 1  0 0 0 0 0 0 1 0 0 0
    */
    // TODO
}