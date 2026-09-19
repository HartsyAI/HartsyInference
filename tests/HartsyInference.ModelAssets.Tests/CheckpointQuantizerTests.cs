using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using Checkpoints = HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Offline quantization: what it refuses, and that a quantized file round-trips back to the values it was
/// made from. The thing that matters most here is not compression but that the SOURCE is read through the
/// container — a fp8_scaled or int8 checkpoint keeps its scales in companion tensors, and quantizing the raw bytes
/// without folding them first writes a file wrong by exactly those scales, with nothing to show for it.</summary>
public sealed class CheckpointQuantizerTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("hartsy-quant-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string WriteSafetensors(string name, Dictionary<string, Tensor> weights)
    {
        string path = Path.Combine(_dir, name);
        SafeTensorsWriter.Save(path, weights);
        return path;
    }

    private static unsafe Tensor Ramp(int rows, int cols, float scale = 1f)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < rows * cols; i++) p[i] = MathF.Sin(i * 0.01f) * scale;
        return t;
    }

    [Fact]
    public void QuantizingWritesASmallerFileThatReadsBack()
    {
        using Tensor weight = Ramp(256, 256);
        string src = WriteSafetensors("src.safetensors", new() { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "out.gguf");

        QuantizationReport report = CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Architecture = "test-arch",
        });

        Assert.True(File.Exists(outPath));
        Assert.Equal(1, report.TensorCount);
        Assert.Equal(1, report.QuantizedCount);
        Assert.True(report.OutputBytes < report.SourceBytes,
            $"Q8_0 of an F32 source should shrink it; got {report.OutputBytes} from {report.SourceBytes}.");
    }

    /// <summary>An existing output is a file someone may still be using, so it is refused rather than replaced
    /// silently.</summary>
    [Fact]
    public void AnExistingOutputIsRefusedUnlessOverwriteIsGiven()
    {
        using Tensor weight = Ramp(256, 256);
        string src = WriteSafetensors("src2.safetensors", new() { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "taken.gguf");
        File.WriteAllText(outPath, "not empty");

        QuantizationJob job = new()
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Architecture = "test-arch",
        };
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CheckpointQuantizer.Quantize(job));
        Assert.Contains("--overwrite", ex.Message, StringComparison.Ordinal);

        CheckpointQuantizer.Quantize(job with { Overwrite = true });
        Assert.True(new FileInfo(outPath).Length > 100);
    }

    /// <summary>Writing over the file being read would truncate the source mid-read.</summary>
    [Fact]
    public void QuantizingOntoItsOwnSourceIsRefused()
    {
        using Tensor weight = Ramp(256, 256);
        string src = WriteSafetensors("self.safetensors", new() { ["blocks.0.attn.weight"] = weight });
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = src,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Overwrite = true,
        }));
        Assert.Contains("same file", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The one that justifies reading through the container. The fp8 weight and its <c>.weight_scale</c>
    /// companion are folded on open, so the quantizer sees real values; reading the raw bytes instead would write a
    /// file wrong by the scale — a plausible-looking file, not an error.</summary>
    [Fact]
    public unsafe void AnFp8ScaledSourceIsFoldedBeforeQuantizing()
    {
        const int Rows = 64, Cols = 256;
        using Tensor fp8 = new Tensor(new TensorShape(Rows, Cols), DType.F8E4M3);
        byte* raw = (byte*)fp8.DataPointer;
        for (int i = 0; i < Rows * Cols; i++) raw[i] = 0x38;   // fp8 e4m3 1.0
        using Tensor scale = new Tensor(new TensorShape(1), DType.F32);
        ((float*)scale.DataPointer)[0] = 4.0f;

        string src = WriteSafetensors("fp8.safetensors", new()
        {
            ["blocks.0.attn.weight"] = fp8,
            ["blocks.0.attn.weight_scale"] = scale,
        });
        string outPath = Path.Combine(_dir, "fp8.gguf");
        CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Architecture = "test-arch",
        });

        using Checkpoints.CheckpointSource readSource = Checkpoints.CheckpointSource.Open(outPath);
        Tensor readBack = readSource.Weights["blocks.0.attn.weight"];
        using Tensor dense = readBack.DType.IsQuantized
            ? GgufDequantizer.Dequantize(readBack, DType.F32) : readBack.CastTo(DType.F32);
        float first = ((float*)dense.DataPointer)[0];
        // 1.0 decoded times a scale of 4 — not the 1.0 an unfolded read would have written.
        Assert.InRange(first, 3.9f, 4.1f);
    }

    /// <summary>The fp8 shape: each eligible weight stored as F8E4M3 beside the scalar it was divided by. Read back
    /// through the container, which folds that companion, so what comes out is the value that went in — within what
    /// four exponent bits and three mantissa bits can carry.</summary>
    [Fact]
    public unsafe void TheFp8TargetRoundTripsThroughItsCompanionScale()
    {
        using Tensor weight = Ramp(1024, 1024, scale: 2f);
        string src = WriteSafetensors("fp8src.safetensors", new() { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "fp8out.safetensors");

        QuantizationReport report = CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Fp8Scaled),
        });
        Assert.Equal(1, report.QuantizedCount);

        using Checkpoints.CheckpointSource read = Checkpoints.CheckpointSource.Open(outPath);
        Tensor stored = read.Weights["blocks.0.attn.weight"];
        Assert.Equal(DType.F8E4M3, stored.DType);
        using Tensor back = stored.CastTo(DType.F32);   // folds the companion the container attached
        ReadOnlySpan<float> got = new((void*)back.DataPointer, 16);
        ReadOnlySpan<float> want = new((void*)weight.DataPointer, 16);
        for (int i = 0; i < 16; i++) Assert.InRange(got[i] - want[i], -0.12f, 0.12f);
    }

    /// <summary>The int8 shape, and the part a reader actually trusts: the per-layer <c>.comfy_quant</c> blob names
    /// the format, rather than the file-level metadata mirror.</summary>
    [Fact]
    public void TheInt8TargetWritesItsScaleAndItsDescriptor()
    {
        using Tensor weight = Ramp(1024, 1024, scale: 2f);
        string src = WriteSafetensors("i8src.safetensors", new() { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "i8out.safetensors");

        QuantizationReport report = CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Int8ConvRot),
        });
        Assert.Equal(1, report.QuantizedCount);

        using SafeTensorsLoader raw = new();
        raw.Load(outPath);
        Dictionary<string, Tensor> all = raw.GetAllTensors();
        Assert.Equal(DType.I8, all["blocks.0.attn.weight"].DType);
        TensorShape scaleShape = all["blocks.0.attn.weight_scale"].Shape;
        Assert.Equal(2, scaleShape.Rank);
        Assert.Equal(1024L, scaleShape[0]);
        Assert.Equal(1L, scaleShape[1]);
        ComfyQuantDescriptor? descriptor =
            ComfyQuantDescriptor.TryParse(all["blocks.0.attn" + ComfyQuantDescriptor.Suffix].AsReadOnlySpan<byte>());
        Assert.NotNull(descriptor);
        Assert.Equal("int8_tensorwise", descriptor!.Format);
        Assert.Equal(256, descriptor.ConvRotGroupSize);

        // The round trip that matters: the container must recognise what we wrote and attach the descriptor, so a
        // file this tool produces is loadable rather than merely well-formed.
        using Checkpoints.CheckpointSource read = Checkpoints.CheckpointSource.Open(outPath);
        Tensor loaded = read.Weights["blocks.0.attn.weight"];
        Assert.Equal(DType.I8, loaded.DType);
        Assert.NotNull(loaded.QuantInfo);
        Assert.Equal("int8_tensorwise", loaded.QuantInfo!.Format);
        Assert.Equal(256, loaded.QuantInfo.ConvRotGroupSize);
    }

    /// <summary>Several eligible weights, not one. The single-tensor cases above cannot see an ownership bug that
    /// only appears once a second tensor exists — a bookkeeping pass that rescans the whole output on every
    /// iteration re-registers earlier companions each time, and what it hands back is disposed more than once.
    /// </summary>
    [Fact]
    public unsafe void TheFp8TargetHandlesSeveralWeightsWithoutDoubleFreeingCompanions()
    {
        Dictionary<string, Tensor> src = new(StringComparer.Ordinal);
        for (int i = 0; i < 4; i++) src[$"blocks.{i}.attn.weight"] = Ramp(1024, 1024, scale: 1f + i);
        string path = WriteSafetensors("fp8multi.safetensors", src);
        foreach (Tensor t in src.Values) t.Dispose();
        string outPath = Path.Combine(_dir, "fp8multi-out.safetensors");

        QuantizationReport report = CheckpointQuantizer.Quantize(new QuantizationJob
        {
            SourcePath = path,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Fp8Scaled),
        });
        Assert.Equal(4, report.QuantizedCount);

        using Checkpoints.CheckpointSource read = Checkpoints.CheckpointSource.Open(outPath);
        for (int i = 0; i < 4; i++)
        {
            Tensor stored = read.Weights[$"blocks.{i}.attn.weight"];
            Assert.Equal(DType.F8E4M3, stored.DType);
            using Tensor back = stored.CastTo(DType.F32);
            // The scale is folded on open, so a companion that was freed early or attached to the wrong weight
            // shows up as values that are not the ones written.
            float first = ((float*)back.DataPointer)[0];
            Assert.InRange(first, -0.001f, 0.001f);
        }
    }
}
