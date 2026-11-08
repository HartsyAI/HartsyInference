using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Engine.Services;

/// <summary>Packs SeedVR2's DiT projections to Q8_0 when the VRAM policy asks for the smaller footprint.</summary>
/// <remarks>Only the 2-D linear weights pack: biases, norms and the embedding tables stay full precision. Q8_0 is one fp16
/// scale per 32 weights, so a packed DiT is about 0.53 of its fp16 size.
/// <para>Measured on an 8 GB RTX 2060 SUPER, 320×240 clip, 9 frames: fp16 streamed 19.8 s; Q8_0 resident 37.3 s. The packed
/// kernel dequantizes each projection to fp16 on every call (SeedVR runs multi-row GEMMs, which the fused packed kernels do
/// not serve), so a packed model is smaller but slower per step than streaming fp16 over PCIe. Auto therefore keeps fp16 and
/// streams it when it does not fit. Q8_0 is the choice the Maximum tier and the <c>numerics.seedvr2Packing=q8</c> knob make.
/// It matches fp16 at 44.9 dB PSNR and 0.9999 SSIM on that clip.</para></remarks>
internal static class SeedVr2WeightPacking
{
    /// <summary>The packing this request runs with: null for the fp16 weights, Q8_0 for packed ones.</summary>
    /// <param name="lever">The resolved <c>QuantizedCompute</c> lever: On packs; Off and Auto keep fp16.</param>
    public static DType? Choose(LeverState lever) => lever == LeverState.On ? DType.Q8_0 : null;

    /// <summary>Replaces each packable linear weight with its packed copy, appending the copies to <paramref name="owned"/> so the
    /// caller releases them with the pipeline. The originals are left alone: their loader owns them.</summary>
    public static void PackLinears(Dictionary<string, Tensor> weights, DType packing, List<Tensor> owned)
    {
        foreach (string key in weights.Keys.ToArray())
        {
            Tensor source = weights[key];
            if (!IsPackable(key, source, packing))
                continue;
            Tensor packed = GgufQuantizer.Quantize(source, packing);
            owned.Add(packed);
            weights[key] = packed;
        }
    }

    /// <summary>A 2-D <c>.weight</c> whose input width is a whole number of packed blocks, outside the timestep embedding.</summary>
    /// <remarks>The embedding projections run on one row per step, the shape the packed kernel mis-serves with F32 activations:
    /// packed, the time embedding comes out wrong and the whole restore is noise (fp16 PSNR-to-reference 9.8 dB, against 56 dB
    /// for the packed blocks alone). They stay fp16; they are under 1% of the model.</remarks>
    internal static bool IsPackable(string key, Tensor tensor, DType packing) =>
        key.EndsWith(".weight", StringComparison.Ordinal)
        && !key.StartsWith("emb_in.", StringComparison.Ordinal)
        && tensor.Shape.Rank == 2
        && tensor.Shape[1] % packing.BlockElementCount == 0;
}
