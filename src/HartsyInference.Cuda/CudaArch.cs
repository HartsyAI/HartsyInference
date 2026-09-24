namespace HartsyInference.Cuda;

/// <summary>Compute capability as one comparable number, <c>major·10 + minor</c>, and the tiers the engine gates on. Every SM check reads these rather than comparing a major/minor pair by hand: a gate written as <c>major >= 12</c> skips SM 10.x, which is how the Blackwell cuBLAS warning missed datacenter parts.</summary>
public static class CudaArch
{
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
}
