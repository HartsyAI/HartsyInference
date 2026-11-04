using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Checkpoints;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// The engine's T5-XXL encoder against the diffusers-style reference written by
/// tests/python-reference/dump_t5_xxl_unmasked_reference.py, loading a standalone T5 file (T5_SAFETENSORS) rather than a Flux
/// checkpoint. Gated: skipped unless T5_SAFETENSORS and T5_REFERENCE_DIR are both set.
/// The reference attends every pad row, as FluxPipeline does, so an all-ones mask must match it within the F32 CPU tolerance.
/// The same encoder run with the real pad mask (the FLUX.1 call site before the fix) is reported as its divergence.
/// </summary>
public sealed class T5StandaloneParityTests
{
    private const float AvgErrTol = 1e-4f; // the F32 CPU tolerance of T5EncoderParityTests: float32 reference, float32 engine

    private readonly ITestOutputHelper _output;

    public T5StandaloneParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void T5XXL_UnmaskedMatchesDiffusersReference_Cpu()
    {
        string? weightsPath = Environment.GetEnvironmentVariable("T5_SAFETENSORS");
        string? referenceDir = Environment.GetEnvironmentVariable("T5_REFERENCE_DIR");
        if (string.IsNullOrEmpty(weightsPath) || string.IsNullOrEmpty(referenceDir) || !File.Exists(Path.Combine(referenceDir, "meta.json")))
        {
            _output.WriteLine("SKIPPED: set T5_SAFETENSORS and T5_REFERENCE_DIR (see tests/python-reference/dump_t5_xxl_unmasked_reference.py).");
            return;
        }

        Meta meta = JsonSerializer.Deserialize<Meta>(File.ReadAllText(Path.Combine(referenceDir, "meta.json")))
            ?? throw new InvalidDataException("T5 reference meta.json malformed.");
        Assert.NotNull(meta.Prompts);

        using CheckpointSource loader = CheckpointSource.Open(weightsPath);
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(loader.Weights);
        using IBackend backend = new CpuBackend();

        foreach (MetaPrompt p in meta.Prompts)
        {
            int[] tokens = ReadInt32(Path.Combine(referenceDir, $"prompt_{p.Index:D2}_tokens.bin"));
            float[] reference = ReadFloat32(Path.Combine(referenceDir, $"prompt_{p.Index:D2}_hidden.bin"));
            int[] unmasked = Enumerable.Repeat(1, tokens.Length).ToArray();
            int[] padMasked = tokens.Select(t => t == 0 ? 0 : 1).ToArray();

            float unmaskedErr = AvgErr(t5.Encode(backend, [tokens], [unmasked]), reference);
            float padMaskedErr = AvgErr(t5.Encode(backend, [tokens], [padMasked]), reference);
            _output.WriteLine($"prompt[{p.Index}] '{p.Prompt}': unmasked avg_err = {unmaskedErr:E3}; pad-masked avg_err = {padMaskedErr:E3} (tolerance {AvgErrTol:E1})");
            Assert.True(unmaskedErr < AvgErrTol, $"prompt[{p.Index}] unmasked avg_err {unmaskedErr:E3} exceeds {AvgErrTol:E1}.");
        }
    }

    private static float AvgErr(Tensor encoded, float[] reference)
    {
        Tensor f32 = encoded.DType == DType.F32 ? encoded : encoded.CastTo(DType.F32);
        try
        {
            return MeanAbsErr(f32, reference);
        }
        finally
        {
            if (!ReferenceEquals(f32, encoded)) f32.Dispose();
            encoded.Dispose();
        }
    }

    private static unsafe float MeanAbsErr(Tensor f32, float[] reference)
    {
        long count = f32.ElementCount;
        if (count != reference.LongLength) throw new ArgumentException($"length mismatch: {count} vs {reference.LongLength}");
        float* ptr = (float*)f32.DataPointer;
        double sum = 0.0;
        for (long i = 0; i < count; i++) sum += Math.Abs(ptr[i] - reference[i]);
        return (float)(sum / count);
    }

    private static int[] ReadInt32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int[] result = new int[raw.Length / sizeof(int)];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }

    private static float[] ReadFloat32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        float[] result = new float[raw.Length / sizeof(float)];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }

    private sealed class Meta
    {
        public string? T5Repo { get; set; }
        public int MaxLength { get; set; }
        public List<MetaPrompt>? Prompts { get; set; }
    }

    private sealed class MetaPrompt
    {
        public int Index { get; set; }
        public string Prompt { get; set; } = string.Empty;
        public int SeqLen { get; set; }
        public int HiddenDim { get; set; }
    }
}
