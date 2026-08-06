using System.Text;
using System.Text.Json;
using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>LoRA merging into the fp8 production checkpoint, which ComfyUI handles by dequantizing, applying the delta
/// in float, and requantizing with a recomputed scale rather than by rejecting the weight. These pin the round trip on
/// decoded values (a scale field that merely changed proves nothing), the determinism that stochastic rounding puts at
/// risk, and the static input scale that a naive replacement would silently zero.</summary>
public sealed class LoraFp8RequantizeTests : IDisposable
{
    private const int Rows = 96, Cols = 64, Rank = 4;

    /// <summary>Real value from the pruned_fp8_scaled checkpoint's <c>out_proj.input_scale</c>.</summary>
    private const float InputScale = 0.0535714291036129f;

    /// <summary>Per-element LoRA delta. Must stay well above the e4m3 quantum: a 4e-4 probe once merged correctly and
    /// still produced bit-identical output, which reads exactly like a broken merge.</summary>
    private const float Delta = 0.05f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "h3-fp8-lora-" + Guid.NewGuid().ToString("N"));

    public LoraFp8RequantizeTests() => Directory.CreateDirectory(_dir);

    /// <summary>The merged weight must decode to base+delta within e4m3 resolution. e4m3 keeps 3 mantissa bits, so the
    /// spacing around a value is v/8 in scaled units — the bound is per-element, not a constant absmax/448 (that is only
    /// the step at the very bottom of the range). Stochastic rounding lands on one of the two bracketing grid points, so
    /// the error is strictly under one ulp.</summary>
    [Fact]
    public void Fp8TargetRequantizesToTheMergedValue()
    {
        string path = QkvLora("fp8merge");
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = Fp8Ramp() };
        float[] baseDecoded = Decode(weights["blocks.0.attn.qkv_proj.weight"]);

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
        Assert.True(merged.DType.IsFp8);
        float[] decoded = Decode(merged);
        Assert.Equal(baseDecoded.Length, decoded.Length);

        float absmax = 0f;
        foreach (float v in decoded)
        {
            if (MathF.Abs(v) > absmax) absmax = MathF.Abs(v);
        }
        Assert.Equal(absmax / 448f, merged.Fp8ScaleFactor, 6);

        double sum = 0.0;
        for (int i = 0; i < decoded.Length; i++)
        {
            float intended = baseDecoded[i] + Delta;
            float error = decoded[i] - intended;
            float tolerance = UlpAt(intended, merged.Fp8ScaleFactor) * 1.001f;
            Assert.True(MathF.Abs(error) <= tolerance,
                $"element {i}: decoded {decoded[i]} vs intended {intended} exceeds one ulp ({tolerance})");
            sum += error;
        }
        // Per-element error is zero-mean by construction; the SEM over these elements is ~5e-4, so this bound flags a
        // systematic offset (a half-ulp bias would be ~0.037) without flaking.
        Assert.True(Math.Abs(sum / decoded.Length) < 5e-3, $"mean signed error {sum / decoded.Length} is biased");
    }

    /// <summary>Stochastic rounding adds a randomized path, so the same merge twice must still be byte-identical or
    /// nothing downstream is reproducible.</summary>
    [Fact]
    public void RequantizedMergeIsBitIdenticalAcrossRuns()
    {
        string path = QkvLora("fp8determinism");
        byte[][] results = new byte[2][];
        float[] scales = new float[2];
        for (int run = 0; run < 2; run++)
        {
            Dictionary<string, Tensor> weights =
                new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = Fp8Ramp() };
            IBackend backend = new CpuBackend();
            using LoraStack stack = new LoraStack();
            stack.AddFromPath(path, strength: 1.0f);
            stack.ApplyTo(weights, LoraTarget.Transformer, backend);
            Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
            results[run] = merged.AsReadOnlySpan<byte>().ToArray();
            scales[run] = merged.Fp8ScaleFactor;
        }
        Assert.Equal(scales[0], scales[1]);
        Assert.True(results[0].AsSpan().SequenceEqual(results[1]), "requantized bytes differ between identical merges");
    }

    /// <summary>A LoRA that targets only non-fp8 modules must be accepted even though the checkpoint holds fp8 tensors —
    /// the old guard scanned the whole dictionary and refused on any of them.</summary>
    [Fact]
    public void LoraTargetingOnlyNonFp8WeightsIsAccepted()
    {
        string path = CreateSafeTensors("fp8selective", new Dictionary<string, (long[] shape, float[] data)>
        {
            ["transformer.blocks.0.mlp.fc2.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, 0.1f)),
            ["transformer.blocks.0.mlp.fc2.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, 0.2f)),
        });
        Tensor untargeted = Fp8Ramp();
        float untargetedScale = untargeted.Fp8ScaleFactor;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["blocks.0.attn.out_proj.weight"] = untargeted,
            ["blocks.0.mlp.fc2.weight"] = new Tensor(new TensorShape(Rows, Cols), DType.BF16),
        };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Equal(DType.BF16, weights["blocks.0.mlp.fc2.weight"].DType);
        Assert.Same(untargeted, weights["blocks.0.attn.out_proj.weight"]);
        Assert.Equal(untargetedScale, untargeted.Fp8ScaleFactor);
    }

    /// <summary>The static activation scale rides on the weight tensor, and the replacement starts at 0, which silently
    /// drops the fused e4m3 modulate path (correct output, far slower). It must be carried across, and it is a separate
    /// value from the weight scale, which is recomputed.</summary>
    [Fact]
    public void MergePreservesTheStaticInputScale()
    {
        string path = QkvLora("fp8inputscale");
        Tensor baseW = Fp8Ramp();
        baseW.Fp8InputScaleFactor = InputScale;
        float originalWeightScale = baseW.Fp8ScaleFactor;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = baseW };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
        Assert.NotSame(baseW, merged);
        Assert.Equal(InputScale, merged.Fp8InputScaleFactor);
        Assert.True(merged.Fp8InputScaleFactor > 0f, "a zeroed input scale silently disables the fused fp8 path");
        Assert.NotEqual(originalWeightScale, merged.Fp8ScaleFactor);
    }

    /// <summary>A per-tensor scale can only be recomputed for a 2-D Linear weight, so anything else must fail by name
    /// rather than corrupt the tensor.</summary>
    [Fact]
    public void NonRank2Fp8TargetThrows()
    {
        string path = QkvLora("fp8rank3");
        Tensor conv = new Tensor(new TensorShape(4, 8, 8), DType.F8E4M3) { Fp8ScaleFactor = 0.01f };
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = conv };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Contains("blocks.0.attn.qkv_proj.weight", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The seeded path must actually round stochastically, not fall through to the cast's round-to-nearest —
    /// the two agree on values already on the grid, so they can only be told apart on values between grid points.</summary>
    [Fact]
    public void SeededQuantizeDiffersFromRoundToNearestButStaysWithinOneUlp()
    {
        byte[] Quantize(string? seedKey)
        {
            Tensor f32 = new Tensor(new TensorShape(Rows, Cols), DType.F32);
            unsafe
            {
                float* p = (float*)f32.DataPointer;
                for (int i = 0; i < Rows * Cols; i++)
                {
                    p[i] = 1.0f + (float)i / (Rows * Cols);
                }
            }
            using Tensor fp8 = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(f32, seedKey);
            f32.Dispose();
            return fp8.AsReadOnlySpan<byte>().ToArray();
        }

        byte[] nearest = Quantize(null);
        byte[] stochastic = Quantize("blocks.0.attn.qkv_proj.weight");
        Assert.False(nearest.AsSpan().SequenceEqual(stochastic), "seeded rounding produced the round-to-nearest result");
        int differing = 0;
        for (int i = 0; i < nearest.Length; i++)
        {
            if (nearest[i] == stochastic[i]) continue;
            differing++;
            // Both must bracket the same value, so their encoded magnitudes are adjacent.
            Assert.True(Math.Abs((nearest[i] & 0x7F) - (stochastic[i] & 0x7F)) == 1,
                $"element {i}: {nearest[i]:X2} and {stochastic[i]:X2} are more than one grid step apart");
        }
        Assert.True(differing > nearest.Length / 10, $"only {differing} of {nearest.Length} elements were perturbed");
    }

    /// <summary>An fp8 [Rows, Cols] weight ramped over [1.0, 2.0] — a narrow range so absmax/448 bounds every element,
    /// and never constant, since a constant fill lands exactly on 448 and would hide a broken rounder.</summary>
    private static Tensor Fp8Ramp()
    {
        int count = Rows * Cols;
        Tensor f32 = new Tensor(new TensorShape(Rows, Cols), DType.F32);
        unsafe
        {
            float* p = (float*)f32.DataPointer;
            for (int i = 0; i < count; i++)
            {
                p[i] = 1.0f + (float)i / count;
            }
        }
        Tensor fp8 = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(f32);
        f32.Dispose();
        return fp8;
    }

    /// <summary>Spacing of the e4m3 grid around <paramref name="value"/>, in real units: 3 mantissa bits below the
    /// binade, then rescaled by the per-tensor factor.</summary>
    private static float UlpAt(float value, float scale)
    {
        float units = MathF.Abs(value) / scale;
        int exponent = (int)MathF.Floor(MathF.Log2(units));
        if (exponent < -6) exponent = -6;
        return MathF.Pow(2f, exponent - 3) * scale;
    }

    /// <summary>Decoded values with the per-tensor scale folded in, which is what the GEMM effectively sees.</summary>
    private static float[] Decode(Tensor fp8)
    {
        using Tensor f32 = fp8.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>A LoRA on the fused qkv projection whose merged delta is exactly <see cref="Delta"/> per element:
    /// alpha defaults to rank so the scale is 1, leaving rank · a · b.</summary>
    private string QkvLora(string name)
    {
        float a = 0.1f;
        float b = Delta / (Rank * a);
        return CreateSafeTensors(name, new Dictionary<string, (long[] shape, float[] data)>
        {
            ["transformer.blocks.0.attn.qkv_proj.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, a)),
            ["transformer.blocks.0.attn.qkv_proj.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, b)),
        });
    }

    private static float[] Filled(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private string CreateSafeTensors(string name, Dictionary<string, (long[] Shape, float[] Data)> tensors)
    {
        using MemoryStream dataStream = new MemoryStream();
        Dictionary<string, (long Start, long End)> offsets = [];
        foreach (KeyValuePair<string, (long[] Shape, float[] Data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float value in kvp.Value.Data)
            {
                dataStream.Write(BitConverter.GetBytes(value), 0, 4);
            }
            offsets[kvp.Key] = (start, dataStream.Position);
        }
        byte[] blob = dataStream.ToArray();

        Dictionary<string, object> header = [];
        foreach (KeyValuePair<string, (long[] Shape, float[] Data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            header[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = DType.F32.Name,
                ["shape"] = kvp.Value.Shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string filePath = Path.Combine(_dir, $"{name}.safetensors");
        using FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new BinaryWriter(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(blob);
        return filePath;
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }
}
