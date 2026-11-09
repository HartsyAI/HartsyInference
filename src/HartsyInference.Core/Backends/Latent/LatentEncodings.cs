namespace HartsyInference.Core.Backends.Latent;

/// <summary>Geometry of each <see cref="LatentEncoding"/>: scale group width and bytes per row.</summary>
public static class LatentEncodings
{
    /// <summary>Elements sharing one scale; 0 for <see cref="LatentEncoding.F32"/>.</summary>
    public static int GroupSize(LatentEncoding encoding) => encoding switch
    {
        LatentEncoding.F32 => 0,
        LatentEncoding.Fp8E4M3Ue8m0x32 => 32,
        LatentEncoding.Fp4E2M1E4M3x16 => 16,
        LatentEncoding.Fp4E2M1E8M0x32 => 32,
        LatentEncoding.Fp8E4M3Ue8m0x128 => 128,
        _ => throw new ArgumentOutOfRangeException(nameof(encoding), encoding, "Unknown latent encoding."),
    };

    /// <summary>True for the two FP4 formats, whose codes pack two elements per byte.</summary>
    public static bool IsFp4(LatentEncoding encoding) =>
        encoding == LatentEncoding.Fp4E2M1E4M3x16 || encoding == LatentEncoding.Fp4E2M1E8M0x32;

    /// <summary>True for the FP8 formats, one e4m3 code byte per element with a UE8M0 group scale.</summary>
    public static bool IsFp8(LatentEncoding encoding) =>
        encoding == LatentEncoding.Fp8E4M3Ue8m0x32 || encoding == LatentEncoding.Fp8E4M3Ue8m0x128;

    /// <summary>Bytes of code storage per row of <paramref name="dim"/> elements.</summary>
    public static long CodeBytesPerRow(LatentEncoding encoding, int dim) => encoding switch
    {
        LatentEncoding.F32 => 4L * dim,
        LatentEncoding.Fp8E4M3Ue8m0x32 or LatentEncoding.Fp8E4M3Ue8m0x128 => dim,
        _ => dim / 2,
    };

    /// <summary>Scale bytes per row of <paramref name="dim"/> elements; 0 for F32.</summary>
    public static int ScaleBytesPerRow(LatentEncoding encoding, int dim)
    {
        int group = GroupSize(encoding);
        return group == 0 ? 0 : dim / group;
    }

    /// <summary>Refuses an encoding the device latent kernels do not decode; they take the encoding as a raw code.</summary>
    /// <param name="operation">The backend and operation, for the message (e.g. "CUDA IndexerScores").</param>
    public static void RequireDeviceKernel(LatentEncoding encoding, string operation)
    {
        if (encoding == LatentEncoding.Fp8E4M3Ue8m0x128)
            throw new NotSupportedException($"{operation} has no kernel for {encoding}; it runs on the CPU backend only.");
    }
}
