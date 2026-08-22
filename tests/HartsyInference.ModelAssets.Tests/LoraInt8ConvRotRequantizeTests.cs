using System.Text;
using System.Text.Json;
using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>LoRA merging into a ComfyUI <c>int8_tensorwise</c> (± ConvRot) checkpoint, which must dequant, merge in
/// float, re-rotate, and requantize with a freshly recomputed per-row absmax/127 scale rather than reject the weight
/// (the BF16-merge shortcut would double the resident DiT and forfeit the IMMA GEMM path). These pin the round trip on
/// decoded values against an independent double-precision Hadamard reference (a scale field that merely changed proves
/// nothing), determinism, the borrowed base staying untouched, and the friendly refusal for an I8 weight with no
/// descriptor.
/// <para><b>Tolerance budget</b> (derived, not borrowed from Int8ConvRotParityTests' 2.5e-2 — that figure leans on
/// quantization-error cancellation a requant cannot claim). Error sources per element, in decode order:
/// (1) requant rounding, at most 0.5·s_new per element in the rotated domain; un-rotation is orthogonal so a group's
/// L2 is preserved, giving a rigorous per-element ceiling of 0.5·√G·s_new[row]; (2) the BF16 hop inside
/// Int8ConvRotCodec.DequantToBf16, whose truncating cast contributes up to one BF16 ulp at the row's magnitude, biased
/// toward zero. The per-element assertion uses the rigorous sum of both; the RMS assertion is the tight tripwire
/// (expected requant RMS is s_new/√12 ≈ 0.29·s_new); the mean assertion bounds systematic bias at one BF16 ulp, small
/// enough that a half-quantum requant bias (0.5·s_new, several times larger here) still trips it.</para></summary>
public sealed unsafe class LoraInt8ConvRotRequantizeTests : IDisposable
{
    private const int Rows = 8, Cols = 64, Rank = 4, GroupSize = 16;

    /// <summary>Per-element LoRA delta. Large enough to move every row's rotated absmax, so a merge that reuses the
    /// old scale clips and fails the round-trip assertions instead of passing by accident.</summary>
    private const float Delta = 0.5f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "int8-convrot-lora-" + Guid.NewGuid().ToString("N"));

    public LoraInt8ConvRotRequantizeTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void ConvRotTargetRequantizesToTheMergedValueWithARecomputedScale()
    {
        double[,] logical = LogicalBase();
        (Tensor baseW, Tensor oldScale, double[] oldScales) = QuantizeReference(logical, GroupSize, fullPrecisionMatMul: true);
        byte[] baseBytes = baseW.AsReadOnlySpan<byte>().ToArray();
        byte[] oldScaleBytes = oldScale.AsReadOnlySpan<byte>().ToArray();
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = baseW };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(QkvLora("int8merge"), strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
        Assert.NotSame(baseW, merged);
        Assert.Equal(DType.I8, merged.DType);
        Assert.Equal(baseW.Shape, merged.Shape);
        QuantWeightInfo info = Assert.IsType<QuantWeightInfo>(merged.QuantInfo);
        Assert.Equal("int8_tensorwise", info.Format);
        Assert.Equal(GroupSize, info.ConvRotGroupSize);
        Assert.True(info.FullPrecisionMatMul, "full_precision_matrix_mult must survive the merge");
        Assert.NotSame(oldScale, info.RowScale);
        Assert.Equal(Rows, info.RowScale!.ElementCount);

        // The borrowed base (mmap-owned in production) must come through untouched.
        Assert.True(baseW.AsReadOnlySpan<byte>().SequenceEqual(baseBytes), "merge mutated the borrowed base weight");
        Assert.True(oldScale.AsReadOnlySpan<byte>().SequenceEqual(oldScaleBytes), "merge mutated the borrowed row scale");

        float[] newScales = info.RowScale.AsReadOnlySpan<float>().ToArray();
        sbyte* q = (sbyte*)merged.DataPointer;
        double sum = 0.0, sumSq = 0.0, scaleSum = 0.0;
        for (int row = 0; row < Rows; row++)
        {
            Assert.NotEqual(oldScales[row], (double)newScales[row]);
            int rowMax = 0;
            for (int c = 0; c < Cols; c++)
            {
                rowMax = Math.Max(rowMax, Math.Abs(q[row * Cols + c]));
            }
            // absmax/127 puts the row's largest magnitude exactly on the grid edge — anything less means the
            // scale was not recomputed from the merged values.
            Assert.Equal(127, rowMax);

            double[] decoded = DecodeRow(q, newScales[row], row, GroupSize);
            double vmax = 0.0;
            for (int c = 0; c < Cols; c++)
            {
                vmax = Math.Max(vmax, Math.Abs(logical[row, c] + Delta));
            }
            double tolerance = 0.5 * Math.Sqrt(GroupSize) * newScales[row] + Bf16UlpAt(vmax);
            for (int c = 0; c < Cols; c++)
            {
                double error = decoded[c] - (logical[row, c] + Delta);
                Assert.True(Math.Abs(error) <= tolerance,
                    $"[{row},{c}]: decoded {decoded[c]} vs intended {logical[row, c] + Delta} exceeds budget {tolerance}");
                sum += error;
                sumSq += error * error;
            }
            scaleSum += newScales[row];
        }
        int count = Rows * Cols;
        double meanScale = scaleSum / Rows;
        double rms = Math.Sqrt(sumSq / count);
        Assert.True(rms <= 0.45 * meanScale + 0.01, $"error RMS {rms} exceeds the tight budget {0.45 * meanScale + 0.01}");
        Assert.True(Math.Abs(sum / count) <= Bf16UlpAt(2.5), $"mean signed error {sum / count} is biased beyond the BF16 hop's truncation");
    }

    [Fact]
    public void UnrotatedInt8TargetRequantizesToTheMergedValue()
    {
        double[,] logical = LogicalBase();
        (Tensor baseW, Tensor _, double[] _) = QuantizeReference(logical, groupSize: 0, fullPrecisionMatMul: false);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = baseW };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(QkvLora("int8plain"), strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
        Assert.Equal(DType.I8, merged.DType);
        QuantWeightInfo info = merged.QuantInfo!;
        Assert.Equal(0, info.ConvRotGroupSize);
        Assert.False(info.FullPrecisionMatMul);

        float[] newScales = info.RowScale!.AsReadOnlySpan<float>().ToArray();
        sbyte* q = (sbyte*)merged.DataPointer;
        for (int row = 0; row < Rows; row++)
        {
            // No rotation ⇒ no √G spread; the budget is a bare half-quantum plus the BF16 hop.
            double tolerance = 0.5 * newScales[row] + Bf16UlpAt(2.5);
            for (int c = 0; c < Cols; c++)
            {
                double decoded = q[row * Cols + c] * (double)newScales[row];
                double error = decoded - (logical[row, c] + Delta);
                Assert.True(Math.Abs(error) <= tolerance,
                    $"[{row},{c}]: decoded {decoded} vs intended {logical[row, c] + Delta} exceeds budget {tolerance}");
            }
        }
    }

    [Fact]
    public void RequantizedMergeIsBitIdenticalAcrossRuns()
    {
        string path = QkvLora("int8determinism");
        byte[][] weightBytes = new byte[2][];
        byte[][] scaleBytes = new byte[2][];
        for (int run = 0; run < 2; run++)
        {
            (Tensor baseW, Tensor _, double[] _) = QuantizeReference(LogicalBase(), GroupSize, fullPrecisionMatMul: false);
            Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = baseW };
            IBackend backend = new CpuBackend();
            using LoraStack stack = new LoraStack();
            stack.AddFromPath(path, strength: 1.0f);
            stack.ApplyTo(weights, LoraTarget.Transformer, backend);
            Tensor merged = weights["blocks.0.attn.qkv_proj.weight"];
            weightBytes[run] = merged.AsReadOnlySpan<byte>().ToArray();
            scaleBytes[run] = merged.QuantInfo!.RowScale!.AsReadOnlySpan<byte>().ToArray();
        }
        Assert.True(weightBytes[0].AsSpan().SequenceEqual(weightBytes[1]), "requantized bytes differ between identical merges");
        Assert.True(scaleBytes[0].AsSpan().SequenceEqual(scaleBytes[1]), "recomputed scales differ between identical merges");
    }

    /// <summary>An I8 weight that lost its descriptor must refuse by name — the old behavior was CastTo's raw
    /// "I8 → F32" HartsyInferenceException with no hint of which weight or why.</summary>
    [Fact]
    public void Int8WeightWithoutQuantInfoRefusesByName()
    {
        Tensor bare = new Tensor(new TensorShape(Rows, Cols), DType.I8);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn.qkv_proj.weight"] = bare };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(QkvLora("int8bare"), strength: 1.0f);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Contains("blocks.0.attn.qkv_proj.weight", ex.Message, StringComparison.Ordinal);
        Assert.Contains("no quantization descriptor", ex.Message, StringComparison.Ordinal);
        bare.Dispose();
    }

    /// <summary>A Comfy full-weight <c>.diff</c> can land on an int8 Linear too, and must ride the same
    /// dequant-merge-requant path as the low-rank pass.</summary>
    [Fact]
    public void FullWeightDiffOntoInt8TargetMerges()
    {
        const float DiffValue = 0.25f;
        string path = CreateSafeTensors("int8diff", new Dictionary<string, (long[] Shape, float[] Data)>
        {
            ["diffusion_model.blocks.0.self_attn.q.diff"] = ([Rows, Cols], Filled(Rows * Cols, DiffValue)),
        });
        double[,] logical = LogicalBase();
        (Tensor baseW, Tensor _, double[] _) = QuantizeReference(logical, GroupSize, fullPrecisionMatMul: false);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { ["blocks.0.attn1.to_q.weight"] = baseW };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Tensor merged = weights["blocks.0.attn1.to_q.weight"];
        Assert.Equal(DType.I8, merged.DType);
        float[] newScales = merged.QuantInfo!.RowScale!.AsReadOnlySpan<float>().ToArray();
        sbyte* q = (sbyte*)merged.DataPointer;
        for (int row = 0; row < Rows; row++)
        {
            double tolerance = 0.5 * Math.Sqrt(GroupSize) * newScales[row] + Bf16UlpAt(2.5);
            double[] decoded = DecodeRow(q, newScales[row], row, GroupSize);
            for (int c = 0; c < Cols; c++)
            {
                double error = decoded[c] - (logical[row, c] + DiffValue);
                Assert.True(Math.Abs(error) <= tolerance,
                    $"[{row},{c}]: decoded {decoded[c]} vs intended {logical[row, c] + DiffValue} exceeds budget {tolerance}");
            }
        }
    }

    /// <summary>Base weights ramped over [1.0, 2.0] with a small deterministic wobble, so no row, group, or element is
    /// constant and every row's absmax moves when <see cref="Delta"/> lands.</summary>
    private static double[,] LogicalBase()
    {
        double[,] logical = new double[Rows, Cols];
        for (int row = 0; row < Rows; row++)
        {
            for (int c = 0; c < Cols; c++)
            {
                int i = row * Cols + c;
                logical[row, c] = 1.0 + (double)i / (Rows * Cols) + 0.05 * ((i * 37 + 11) % 17) / 17.0;
            }
        }
        return logical;
    }

    /// <summary>Quantizes the logical weights into ConvRot storage (rotate each group by the reference H, then
    /// per-row absmax/127) and returns the packed I8 tensor with its QuantInfo attached, plus the raw scales.</summary>
    private static (Tensor Weight, Tensor RowScale, double[] Scales) QuantizeReference(
        double[,] logical, int groupSize, bool fullPrecisionMatMul)
    {
        Tensor weight = new Tensor(new TensorShape(Rows, Cols), DType.I8);
        Tensor rowScale = new Tensor(new TensorShape(Rows), DType.F32);
        double[] scales = new double[Rows];
        sbyte* q = (sbyte*)weight.DataPointer;
        float* s = (float*)rowScale.DataPointer;
        for (int row = 0; row < Rows; row++)
        {
            double[] rotated = new double[Cols];
            for (int c = 0; c < Cols; c++)
            {
                rotated[c] = logical[row, c];
            }
            if (groupSize > 0)
            {
                rotated = RotateReference(rotated, groupSize);
            }
            double absmax = 0.0;
            for (int c = 0; c < Cols; c++)
            {
                absmax = Math.Max(absmax, Math.Abs(rotated[c]));
            }
            double scale = absmax / 127.0;
            scales[row] = scale;
            s[row] = (float)scale;
            for (int c = 0; c < Cols; c++)
            {
                q[row * Cols + c] = (sbyte)Math.Clamp((int)Math.Round(rotated[c] / scale), -127, 127);
            }
        }
        weight.QuantInfo = new QuantWeightInfo
        {
            Format = "int8_tensorwise",
            RowScale = rowScale,
            ConvRotGroupSize = groupSize,
            FullPrecisionMatMul = fullPrecisionMatMul,
        };
        return (weight, rowScale, scales);
    }

    /// <summary>Decodes one stored row back to logical values: int8 · scale, then the same symmetric-orthogonal H
    /// un-rotates (H·H = I), all in double via the independent reference matrix.</summary>
    private static double[] DecodeRow(sbyte* q, float scale, int row, int groupSize)
    {
        double[] values = new double[Cols];
        for (int c = 0; c < Cols; c++)
        {
            values[c] = q[row * Cols + c] * (double)scale;
        }
        return groupSize > 0 ? RotateReference(values, groupSize) : values;
    }

    /// <summary>v @ H per contiguous group, using <see cref="ReferenceHadamard"/> — deliberately NOT the codec's
    /// butterfly (or BuildHadamard, which the butterfly generates), so the reference is independent of the code under
    /// test.</summary>
    private static double[] RotateReference(double[] values, int groupSize)
    {
        double[,] h = ReferenceHadamard(groupSize);
        double[] result = new double[values.Length];
        for (int start = 0; start < values.Length; start += groupSize)
        {
            for (int j = 0; j < groupSize; j++)
            {
                double acc = 0.0;
                for (int i = 0; i < groupSize; i++)
                {
                    acc += values[start + i] * h[i, j];
                }
                result[start + j] = acc;
            }
        }
        return result;
    }

    /// <summary>comfy-kitchen's normalized regular Hadamard as an explicit Kronecker power of the 4×4 seed:
    /// H[i,j] = ∏_d h4[i_d, j_d] / 2^digits over base-4 digits, each factor carrying the seed's 1/2 normalization.</summary>
    private static double[,] ReferenceHadamard(int size)
    {
        int[,] h4 = new int[4, 4] { { 1, 1, 1, -1 }, { 1, 1, -1, 1 }, { 1, -1, 1, 1 }, { -1, 1, 1, 1 } };
        int digits = 0;
        for (int v = size; v > 1; v /= 4)
        {
            digits++;
        }
        double norm = Math.Pow(0.5, digits);
        double[,] h = new double[size, size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int product = 1, a = i, b = j;
                for (int d = 0; d < digits; d++)
                {
                    product *= h4[a & 3, b & 3];
                    a >>= 2;
                    b >>= 2;
                }
                h[i, j] = product * norm;
            }
        }
        return h;
    }

    /// <summary>BF16 ulp at <paramref name="value"/>: 7 stored mantissa bits below the binade. The codec's F32→BF16
    /// cast truncates, so a full ulp (not half) is the per-element ceiling for that hop.</summary>
    private static double Bf16UlpAt(double value) =>
        Math.Pow(2.0, Math.Floor(Math.Log2(Math.Abs(value))) - 7);

    /// <summary>A LoRA on the qkv projection whose merged delta is exactly <see cref="Delta"/> per element:
    /// alpha defaults to rank so the scale is 1, leaving rank · a · b.</summary>
    private string QkvLora(string name)
    {
        float a = 0.1f;
        float b = Delta / (Rank * a);
        return CreateSafeTensors(name, new Dictionary<string, (long[] Shape, float[] Data)>
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
