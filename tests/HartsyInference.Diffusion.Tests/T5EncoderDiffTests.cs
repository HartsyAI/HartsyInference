using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Numerical validation gate for T5-XXL: encodes the prompts dumped by `tests/python-reference/dump_t5_xxl_hidden_states.py` and asserts the C# hidden states match the HuggingFace transformers reference within F32 noise. Skips cleanly when the reference dump or any prerequisite asset is missing.
///
/// <para>Tolerance: <see cref="CpuAvgErrTol"/> for CPU (1e-5 F32 reference vs F32 implementation), <see cref="GpuAvgErrTol"/> for GPU (slightly looser to absorb F32 → F16 rounding through cuBLAS GEMM).</para></summary>
[Trait("Category", "Integration")]
public sealed class T5EncoderDiffTests
{
    private const float CpuAvgErrTol = 1e-4f;
    private const float GpuAvgErrTol = 1e-3f;

    private static string ReferenceDir =>
        Environment.GetEnvironmentVariable("T5_REFERENCE_DIR")
        ?? Path.Combine(RepoRoot.Path, "tests", "python-reference", "t5_reference");

    private readonly ITestOutputHelper _output;
    public T5EncoderDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void T5XXL_MatchesPythonReference_Cpu() => Run(useCuda: false, tol: CpuAvgErrTol);

    [Fact]
    public void T5XXL_MatchesPythonReference_Gpu() => Run(useCuda: true, tol: GpuAvgErrTol);

    private void Run(bool useCuda, float tol)
    {
        string metaPath = Path.Combine(ReferenceDir, "meta.json");
        if (!File.Exists(metaPath))
        {
            _output.WriteLine($"SKIPPED: T5 reference dir not found at {ReferenceDir}.");
            _output.WriteLine("Generate references first via: python tests/python-reference/dump_t5_xxl_hidden_states.py --output <dir>");
            return;
        }
        if (!File.Exists(TestPaths.Flux.Dev) && !File.Exists(TestPaths.Flux.Schnell))
        {
            _output.WriteLine("SKIPPED: no Flux checkpoint to source T5 weights from.");
            return;
        }
        if (useCuda && !CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }

        Meta meta = JsonSerializer.Deserialize<Meta>(File.ReadAllText(metaPath))
            ?? throw new InvalidDataException("T5 reference meta.json malformed.");
        if (meta.Prompts is null || meta.Prompts.Count == 0)
        {
            _output.WriteLine("SKIPPED: no prompts in T5 reference meta.json.");
            return;
        }

        string sourceCheckpoint = File.Exists(TestPaths.Flux.Dev) ? TestPaths.Flux.Dev : TestPaths.Flux.Schnell;
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(sourceCheckpoint);
        sw.Stop();
        _output.WriteLine($"Loaded T5 weights from {Path.GetFileName(sourceCheckpoint)} in {sw.ElapsedMilliseconds}ms.");

        using (loader)
        {
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);

            string? ptxDir = useCuda ? Path.Combine(RepoRoot.Path, "native", "cuda", "build") : null;
            if (useCuda && (ptxDir is null || !Directory.Exists(ptxDir)))
            {
                _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
                return;
            }

            using IBackend backend = useCuda
                ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir!)
                : new CpuBackend();

            int prompts = Math.Min(meta.Prompts.Count, 3);
            float aggregateAvg = 0f;
            int countedPrompts = 0;
            for (int idx = 0; idx < prompts; idx++)
            {
                MetaPrompt p = meta.Prompts[idx];
                string tokensFile = Path.Combine(ReferenceDir, $"prompt_{idx:D2}_tokens.bin");
                string maskFile = Path.Combine(ReferenceDir, $"prompt_{idx:D2}_attention_mask.bin");
                string hiddenFile = Path.Combine(ReferenceDir, $"prompt_{idx:D2}_hidden.bin");
                if (!File.Exists(tokensFile) || !File.Exists(maskFile) || !File.Exists(hiddenFile))
                {
                    _output.WriteLine($"  prompt[{idx}]: skipped (binaries missing)");
                    continue;
                }

                int[] tokenIds = ReadInt32(tokensFile);
                int[] mask = ReadInt32(maskFile);
                float[] referenceHidden = ReadFloat32(hiddenFile);
                int seqLen = tokenIds.Length;
                int hiddenDim = p.HiddenDim;
                long expected = (long)seqLen * hiddenDim;
                if (referenceHidden.Length != expected)
                {
                    _output.WriteLine($"  prompt[{idx}]: skipped (hidden length {referenceHidden.Length} != {expected})");
                    continue;
                }

                Tensor encoded = t5.Encode(backend, [tokenIds], [mask]);
                Assert.Equal(3, encoded.Shape.Rank);
                Assert.Equal(seqLen, (int)encoded.Shape[1]);
                Assert.Equal(hiddenDim, (int)encoded.Shape[2]);

                Tensor encodedF32 = encoded.DType == DType.F32 ? encoded : encoded.CastTo(DType.F32);
                if (!ReferenceEquals(encodedF32, encoded)) encoded.Dispose();

                float avgErr = MeanAbsErr(encodedF32, referenceHidden);
                _output.WriteLine($"  prompt[{idx}] '{p.Prompt[..Math.Min(40, p.Prompt.Length)]}…' avg_err = {avgErr:E3}");
                encodedF32.Dispose();

                aggregateAvg += avgErr;
                countedPrompts++;
            }

            if (countedPrompts == 0)
            {
                _output.WriteLine("SKIPPED: no usable prompts in reference dir.");
                return;
            }

            float meanAvgErr = aggregateAvg / countedPrompts;
            _output.WriteLine($"Mean avg_err across {countedPrompts} prompts: {meanAvgErr:E3} (tol: {tol:E3}).");
            Assert.True(meanAvgErr < tol, $"T5 mean avg_err {meanAvgErr:E3} exceeds tolerance {tol:E3}.");

            t5.Dispose();
        }
    }

    private static int[] ReadInt32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length % 4 != 0) throw new InvalidDataException($"{path}: not a multiple of 4 bytes.");
        int[] result = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }

    private static float[] ReadFloat32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length % 4 != 0) throw new InvalidDataException($"{path}: not a multiple of 4 bytes.");
        float[] result = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }

    private static unsafe float MeanAbsErr(Tensor f32, float[] reference)
    {
        if (f32.DType != DType.F32) throw new ArgumentException("expected F32", nameof(f32));
        long count = f32.ElementCount;
        if (count != reference.LongLength) throw new ArgumentException($"length mismatch: {count} vs {reference.LongLength}");
        float* ptr = (float*)f32.DataPointer;
        double sum = 0.0;
        for (long i = 0; i < count; i++) sum += Math.Abs(ptr[i] - reference[i]);
        return (float)(sum / count);
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
