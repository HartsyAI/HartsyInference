namespace HartsyInference.Cuda;

/// <summary>Compute capability as one comparable number, <c>major·10 + minor</c>, and the tiers the engine gates on. Every SM check reads these rather than comparing a major/minor pair by hand: a gate written as <c>major >= 12</c> skips SM 10.x, which is how the Blackwell cuBLAS warning missed datacenter parts.</summary>
public static class CudaArch
{
    /// <summary>SM 7.5 — Turing (RTX 20xx, GTX 16xx). Has F16 tensor cores (m16n8k8) but no <c>cp.async</c>, no m16n8k16 MMA
    /// and no BF16 tensor cores; the floor the CUDA backend can run on.</summary>
    public const int Turing = 75;

    /// <summary>SM 8.0 — TF32, BF16, the tensor-core GEMM and VSA floors.</summary>
    public const int Ampere = 80;

    /// <summary>SM 8.9 — native FP8 tensor cores.</summary>
    public const int Ada = 89;

    /// <summary>SM 9.0.</summary>
    public const int Hopper = 90;

    /// <summary>SM 10.0 — block-scaled FP4/MXFP8 tensor cores (datacenter B200/B300).</summary>
    public const int Blackwell = 100;

    /// <summary>SM 12.0 — consumer Blackwell (RTX 50xx): the same block-scaled formats, no tcgen05.</summary>
    public const int BlackwellConsumer = 120;

    public static int Sm(int major, int minor) => major * 10 + minor;

    // Feature predicates over a compute capability. Each names the hardware feature a gate depends on, so a gate reads as what
    // it needs rather than as a bare SM comparison; the thresholds are the tiers above.

    /// <summary>The m16n8k16 F16 MMA and <c>cp.async</c> (the Ampere tensor-core GEMM and FA2 causal prefill). Turing has
    /// neither, so the kernels that use them are built as sm_80 PTX. Video sparse attention is gated here too, though its
    /// kernel needs neither; that gate keeps <c>h3_vsa.sm75.ptx</c> off Turing until a Turing run verifies it.</summary>
    public static bool HasMmaK16(int sm) => sm >= Ampere;

    /// <summary>TF32 tensor-core math for F32 GEMMs.</summary>
    public static bool HasTf32(int sm) => sm >= Ampere;

    /// <summary>BF16 tensor math and BF16 weights at full speed. Turing computes BF16 only in software.</summary>
    public static bool HasBf16(int sm) => sm >= Ampere;

    /// <summary>Native FP8 tensor cores (cuBLASLt FP8 GEMM).</summary>
    public static bool HasFp8TensorCores(int sm) => sm >= Ada;

    /// <summary>Block-scaled FP4/MXFP8 tensor cores.</summary>
    public static bool HasFp4TensorCores(int sm) => sm >= Blackwell;
}
