namespace HartsyInference.Core.Backends;

/// <summary>Storage format of a latent cache row; every quantized format scales fixed groups along the row.</summary>
public enum LatentEncoding
{
    /// <summary>Plain F32, no scales.</summary>
    F32 = 0,

    /// <summary>FP8 e4m3 codes, one byte each, with a UE8M0 scale byte per 32 elements (sliding-window ring).</summary>
    Fp8E4M3Ue8m0x32 = 1,

    /// <summary>FP4 e2m1 codes packed two per byte (even element low), with an e4m3 scale byte per 16 (compressed cache).</summary>
    Fp4E2M1E4M3x16 = 2,

    /// <summary>FP4 e2m1 codes packed two per byte (even element low), with a UE8M0 scale byte per 32 (indexer keys).</summary>
    Fp4E2M1E8M0x32 = 3,
}
