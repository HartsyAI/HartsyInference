using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Mxfp4;

/// <summary>Dequantizer for MXFP4-packed weights (OpenAI GPT-OSS / Microsoft Lens text encoder format).
///
/// <para><b>MXFP4 format:</b> 4-bit floating-point values from a fixed 16-entry sign-magnitude lookup table
/// (8 magnitudes, sign bit), packed two-per-byte (low nibble = even index, high nibble = odd index in the
/// dequantized last dimension). Each contiguous block of 32 dequantized elements shares one E8M0 scale —
/// an 8-bit unsigned integer interpreted as a biased exponent: <c>actual_exp = stored - 127</c>, with the
/// dequant value computed as <c>fp4_lookup[nibble] * 2^actual_exp</c>.</para>
///
/// <para><b>Tensor naming convention</b> in the upstream HuggingFace transformers integration: each
/// MXFP4-quantized projection has two companion safetensors keys, <c>{name}_blocks</c> (uint8) and
/// <c>{name}_scales</c> (uint8/int8). For example, GPT-OSS' MoE experts ship as
/// <c>model.layers.{i}.mlp.experts.gate_up_proj_blocks</c> + <c>...gate_up_proj_scales</c>,
/// dequantizing to <c>...gate_up_proj</c> shape <c>[numExperts, hidden, 2*intermediate]</c>.</para>
///
/// <para><b>Shape relationship:</b> if <c>blocks</c> has shape <c>[..., N]</c> (uint8 bytes) and the
/// dequantized tensor should have shape <c>[..., 2N]</c> (F32 elements), then <c>scales</c> has shape
/// <c>[..., 2N / 32]</c> = <c>[..., N / 16]</c> (one scale per 32-element block = 16-byte chunk).
/// All three tensors share a leading prefix shape; only the trailing dim differs.</para>
///
/// Reference: <c>transformers/integrations/mxfp4.py</c>.</summary>
public static unsafe class Mxfp4Codec
{
    /// <summary>The 16-entry FP4 lookup table. Index = 4-bit value (0-15); sign bit is bit-3
    /// (indices 8-15 are negative). Magnitude values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0.</summary>
    public static readonly float[] Fp4Lut =
    [
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    /// <summary>Number of FP4 elements per E8M0 scale block. Hard-coded in upstream.</summary>
    public const int BlockSize = 32;

    /// <summary>E8M0 bias offset — <c>stored = bias + actual_exponent</c>.</summary>
    public const int E8M0Bias = 127;

    /// <summary>Dequantizes a paired (<paramref name="blocks"/>, <paramref name="scales"/>) MXFP4 weight
    /// into a fresh F32 <see cref="Tensor"/>. Output element count is <c>blocks.ElementCount * 2</c>; the
    /// caller supplies <paramref name="dequantShape"/> to determine the output's logical shape, which must
    /// match that element count.
    /// <para><b>Layout assumption:</b> bytes in <paramref name="blocks"/> are linear-row-major; the low
    /// nibble of byte <c>j</c> dequantizes to output position <c>2j</c>, the high nibble to <c>2j + 1</c>.
    /// One scale covers each consecutive 16-byte (32-element) block in linear order. This matches the
    /// upstream packing used by <c>transformers/integrations/mxfp4.py</c>.</para></summary>
    /// <param name="blocks">MXFP4-packed bytes (one byte per two FP4 values). Must be U8.</param>
    /// <param name="scales">E8M0 per-block exponent table. Must be U8.</param>
    /// <param name="dequantShape">Logical shape of the output. <c>ElementCount</c> must equal
    /// <c>blocks.ElementCount * 2</c>.</param>
    public static Tensor DequantToF32(Tensor blocks, Tensor scales, TensorShape dequantShape)
    {
        if (blocks.DType != DType.U8)
            throw new ArgumentException($"blocks must be U8; got {blocks.DType}.", nameof(blocks));
        if (scales.DType != DType.U8)
            throw new ArgumentException($"scales must be U8; got {scales.DType}.", nameof(scales));

        long byteCount = blocks.Shape.ElementCount;
        long scaleCount = scales.Shape.ElementCount;
        long expectedElements = byteCount * 2;
        if (dequantShape.ElementCount != expectedElements)
            throw new ArgumentException(
                $"dequantShape elementCount {dequantShape.ElementCount} must equal blocks.ElementCount * 2 = {expectedElements}.",
                nameof(dequantShape));
        long expectedBlocks = expectedElements / BlockSize;
        if (expectedElements % BlockSize != 0)
            throw new ArgumentException(
                $"dequantShape elementCount {expectedElements} must be a multiple of block size {BlockSize}.",
                nameof(dequantShape));
        if (scaleCount != expectedBlocks)
            throw new ArgumentException(
                $"scales.ElementCount {scaleCount} must equal expected block count {expectedBlocks} (1 scale per {BlockSize} elements).",
                nameof(scales));

        Tensor output = new Tensor(dequantShape, DType.F32);
        byte* bytesPtr = (byte*)blocks.DataPointer;
        byte* scalesPtr = (byte*)scales.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Block-by-block dequant. Each block: 16 bytes → 32 F32 values; one shared scale.
        long bytesPerBlock = BlockSize / 2;
        for (long b = 0; b < expectedBlocks; b++)
        {
            int expRaw = scalesPtr[b];
            int actualExp = expRaw - E8M0Bias;
            float blockScale = Pow2(actualExp);

            long byteStart = b * bytesPerBlock;
            long outStart = b * BlockSize;
            for (long j = 0; j < bytesPerBlock; j++)
            {
                byte packed = bytesPtr[byteStart + j];
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                outPtr[outStart + 2 * j] = Fp4Lut[lo] * blockScale;
                outPtr[outStart + 2 * j + 1] = Fp4Lut[hi] * blockScale;
            }
        }

        return output;
    }

    /// <summary>Walks an in-memory weight dict, finds every <c>{name}_blocks</c> / <c>{name}_scales</c>
    /// companion pair, dequantizes each into a fresh F32 tensor stored under the bare <c>{name}</c> key,
    /// and removes the companion pairs. Idempotent for keys that have no companion. Mirrors the way
    /// transformers' MXFP4 loader rewrites the state_dict before assigning to model parameters.
    /// <para>The dequantized shape is inferred from <paramref name="shapeOracle"/> when supplied; if
    /// no oracle is given, the function assumes the dequant shape is <c>blocks.Shape</c> with its last
    /// dim doubled (the common case for the GPT-OSS expert layout — the last byte axis decompresses
    /// to twice as many FP4 elements).</para></summary>
    /// <param name="weights">Mutable dict of named tensors. Companion pairs are removed; dequant'd
    /// tensors are added under the bare name.</param>
    /// <param name="shapeOracle">Optional callback: given the bare name, returns the desired
    /// dequantized shape. Return <c>null</c> to fall back to <c>blocks.Shape with last-dim doubled</c>.</param>
    /// <returns>Number of companion pairs dequantized.</returns>
    public static int DequantAllPairsInPlace(Dictionary<string, Tensor> weights,
        Func<string, TensorShape?>? shapeOracle = null)
    {
        List<string> blocksKeys = new();
        foreach (string key in weights.Keys)
            if (key.EndsWith("_blocks", StringComparison.Ordinal))
                blocksKeys.Add(key);

        int dequanted = 0;
        foreach (string blocksKey in blocksKeys)
        {
            string baseName = blocksKey[..^"_blocks".Length];
            string scalesKey = $"{baseName}_scales";
            if (!weights.TryGetValue(scalesKey, out Tensor? scales)) continue;
            Tensor blocks = weights[blocksKey];

            TensorShape dequantShape = shapeOracle?.Invoke(baseName) ?? DoubleLastDim(blocks.Shape);
            Tensor dequanted_ = DequantToF32(blocks, scales, dequantShape);
            weights[baseName] = dequanted_;
            weights.Remove(blocksKey);
            weights.Remove(scalesKey);
            blocks.Dispose();
            scales.Dispose();
            dequanted++;
        }
        return dequanted;
    }

    /// <summary>Dequantizes a GPT-OSS Mixture-of-Experts weight (<c>experts.gate_up_proj</c> /
    /// <c>experts.down_proj</c>) from its canonical 4D MXFP4 on-disk layout, reproducing the runtime
    /// parameter shape that <see cref="SharpInference.Diffusion.Models.TextEncoders.GptOssMoeFfn"/> expects.
    /// <para><b>Layout:</b> on disk the blocks tensor is <c>[E, A, G, 16]</c> (U8) with a companion
    /// scales tensor <c>[E, A, G]</c> (U8, E8M0). Each row of 16 bytes dequantizes to 32 FP4 values along
    /// an implicit within-block axis; the <c>G</c> blocks concatenate to a length-<c>G·32</c> axis. The
    /// upstream <c>convert_moe_packed_tensors</c> then <b>transposes the last two axes</b>, so the runtime
    /// parameter is <c>[E, G·32, A]</c> — i.e. <c>gate_up_proj</c> becomes <c>[E, hidden, 2·intermediate]</c>
    /// and <c>down_proj</c> becomes <c>[E, intermediate, hidden]</c>. This method bakes that transpose into
    /// the dequant so the output is directly forward-pass-ready. Verified byte-exact against
    /// <c>transformers.integrations.mxfp4.convert_moe_packed_tensors</c>.</para></summary>
    /// <param name="blocks">MXFP4-packed bytes, shape <c>[E, A, G, 16]</c>, dtype U8.</param>
    /// <param name="scales">E8M0 per-block exponents, shape <c>[E, A, G]</c>, dtype U8.</param>
    /// <returns>Dequantized F32 tensor of shape <c>[E, G·32, A]</c>.</returns>
    public static Tensor DequantGptOssExpert(Tensor blocks, Tensor scales)
    {
        if (blocks.DType != DType.U8)
            throw new ArgumentException($"blocks must be U8; got {blocks.DType}.", nameof(blocks));
        if (scales.DType != DType.U8)
            throw new ArgumentException($"scales must be U8; got {scales.DType}.", nameof(scales));
        if (blocks.Shape.Rank != 4)
            throw new ArgumentException(
                $"GPT-OSS expert blocks must be rank-4 [E, A, G, 16]; got rank {blocks.Shape.Rank} ({blocks.Shape}).",
                nameof(blocks));
        if (scales.Shape.Rank != 3)
            throw new ArgumentException(
                $"GPT-OSS expert scales must be rank-3 [E, A, G]; got rank {scales.Shape.Rank} ({scales.Shape}).",
                nameof(scales));

        long E = blocks.Shape[0];
        long A = blocks.Shape[1];
        long G = blocks.Shape[2];
        long bytesPerBlock = blocks.Shape[3];
        if (bytesPerBlock != BlockSize / 2)
            throw new ArgumentException(
                $"GPT-OSS expert blocks last dim must be {BlockSize / 2} bytes (one MXFP4 block = {BlockSize} elements); got {bytesPerBlock}.",
                nameof(blocks));
        if (scales.Shape[0] != E || scales.Shape[1] != A || scales.Shape[2] != G)
            throw new ArgumentException(
                $"scales shape {scales.Shape} must equal blocks.shape[:-1] = [{E}, {A}, {G}].", nameof(scales));

        long hidden = G * BlockSize;  // G·32
        Tensor output = new Tensor(new TensorShape(E, hidden, A), DType.F32);
        byte* bytesPtr = (byte*)blocks.DataPointer;
        byte* scalesPtr = (byte*)scales.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (long e = 0; e < E; e++)
        {
            for (long a = 0; a < A; a++)
            {
                for (long g = 0; g < G; g++)
                {
                    long scaleIdx = (e * A + a) * G + g;
                    float blockScale = Pow2(scalesPtr[scaleIdx] - E8M0Bias);
                    long byteBase = scaleIdx * bytesPerBlock;
                    long hiddenBase = g * BlockSize;
                    for (long j = 0; j < bytesPerBlock; j++)
                    {
                        byte packed = bytesPtr[byteBase + j];
                        int lo = packed & 0x0F;
                        int hi = (packed >> 4) & 0x0F;
                        long h0 = hiddenBase + 2 * j;
                        // Transposed write: output[e, h, a] at linear (e·hidden + h)·A + a.
                        outPtr[(e * hidden + h0) * A + a] = Fp4Lut[lo] * blockScale;
                        outPtr[(e * hidden + h0 + 1) * A + a] = Fp4Lut[hi] * blockScale;
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Walks a GPT-OSS encoder weight dict and dequantizes every MoE expert MXFP4 companion pair
    /// (<c>…experts.gate_up_proj_blocks</c>/<c>_scales</c> and <c>…experts.down_proj_blocks</c>/<c>_scales</c>)
    /// via <see cref="DequantGptOssExpert"/>, storing the forward-ready F32 tensor under the bare name and
    /// removing the companions. Idempotent for dicts with no MXFP4 pairs (e.g. an already-dequantized or
    /// BF16 checkpoint). Use this before handing weights to the encoder loader.</summary>
    /// <param name="weights">Mutable named-tensor dict (HuggingFace <c>model.*</c> naming).</param>
    /// <returns>Number of expert pairs dequantized.</returns>
    public static int DequantGptOssExpertsInPlace(Dictionary<string, Tensor> weights)
    {
        List<string> blocksKeys = new();
        foreach (string key in weights.Keys)
            if (key.EndsWith("experts.gate_up_proj_blocks", StringComparison.Ordinal) ||
                key.EndsWith("experts.down_proj_blocks", StringComparison.Ordinal))
                blocksKeys.Add(key);

        int dequanted = 0;
        foreach (string blocksKey in blocksKeys)
        {
            string baseName = blocksKey[..^"_blocks".Length];
            string scalesKey = $"{baseName}_scales";
            if (!weights.TryGetValue(scalesKey, out Tensor? scales))
                throw new InvalidOperationException(
                    $"MXFP4 blocks key '{blocksKey}' has no companion scales key '{scalesKey}'.");
            Tensor blocks = weights[blocksKey];

            Tensor expert = DequantGptOssExpert(blocks, scales);
            weights[baseName] = expert;
            weights.Remove(blocksKey);
            weights.Remove(scalesKey);
            blocks.Dispose();
            scales.Dispose();
            dequanted++;
        }
        return dequanted;
    }

    private static TensorShape DoubleLastDim(TensorShape shape)
    {
        int rank = shape.Rank;
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++) dims[i] = shape[i];
        dims[rank - 1] *= 2;
        return new TensorShape(dims);
    }

    /// <summary>Computes <c>2^n</c> using fast bit-twiddling for the IEEE-754 single-precision exponent
    /// range. For <c>n</c> outside <c>[-127, 127]</c> falls back to <see cref="MathF.Pow"/>.</summary>
    private static float Pow2(int n)
    {
        if (n >= -126 && n <= 127)
        {
            uint bits = (uint)((n + 127) << 23);
            return BitConverter.UInt32BitsToSingle(bits);
        }
        return MathF.Pow(2.0f, n);
    }
}
