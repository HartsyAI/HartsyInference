using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests.Parity;

/// <summary>Encodes a real prompt with the REAL LTX-2.5 Gemma-4-12B text encoder in its
/// <c>int8_tensorwise</c> + ConvRot build — the only text encoder Lightricks ships small enough to run here
/// (15.37 GB against the bf16 build's 26.26 GB).</summary>
/// <remarks><para>This is the intersection of two independently-verified pieces that had never met: the Gemma 4
/// port (checked on a tiny synthetic config) and the resident int8 path (checked on a video DiT). 328 of the
/// encoder's Linears are packed int8, so every projection in every block runs through the INT8 IMMA path — if
/// the ConvRot rotation or the per-row scale were wrong, the residual stream would blow up or collapse within a
/// few blocks rather than produce a usable conditioning state.</para>
/// <para>Sanity gates, not parity gates: there is no reference dump for this checkpoint. What is asserted is
/// what a broken quantized encoder cannot fake — finite states, a residual scale in the band a working Gemma
/// produces, and prompt sensitivity.</para></remarks>
[Collection("CudaSerial")]
public sealed class Gemma4Int8ConvRotEncodeTests(ITestOutputHelper output)
{
    private const string FileName = "gemma4-12b-with-proj-ltx-2.5-int8_lean_convrot.safetensors";

    /// <summary>Declared size of the build (15.37 GB); an in-progress download passes <see cref="File.Exists"/>.</summary>
    private const long MinimumBytes = 15_000_000_000L;

    private readonly ITestOutputHelper _output = output;

    private static string? CheckpointPath()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("LTX25_GEMMA4_INT8");
        if (!string.IsNullOrWhiteSpace(fromEnv)) return Complete(fromEnv) ? fromEnv : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "Models", "text_encoders", FileName);
            if (Complete(candidate)) return candidate;
        }
        return null;
    }

    private static bool Complete(string path) => File.Exists(path) && new FileInfo(path).Length >= MinimumBytes;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public unsafe void EncodesARealPromptFromThePackedInt8Checkpoint()
    {
        string? path = CheckpointPath();
        if (path is null) return;   // tier-lint: guarded
        if (!CudaContext.IsAvailable()) return;   // tier-lint: guarded

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        IReadOnlyDictionary<string, Tensor> raw = loader.GetAllTensors();

        int packed = 0;
        foreach (KeyValuePair<string, Tensor> entry in raw)
        {
            if (entry.Value.DType == DType.I8 && entry.Key.EndsWith(".weight", StringComparison.Ordinal)) packed++;
        }
        _output.WriteLine($"int8 weights in the file: {packed}");
        Assert.Equal(328, packed);

        // The embedding table is deliberately left BF16 by the quantizer, which is why no int8 embedding-gather
        // path is needed. Assert it, because an int8 embed_tokens would silently need one.
        Assert.Equal(DType.BF16, raw["model.embed_tokens.weight"].DType);

        using Gemma4TextEncoder encoder = new Gemma4TextEncoder(Gemma4TextEncoderConfig.Gemma4_12B);
        encoder.LoadWeights(raw);

        Gemma4Tokenizer tokenizer = Gemma4Tokenizer.FromTokenizerJson(
            new ReadOnlySpan<byte>(raw["tokenizer_json"].DataPointer, (int)raw["tokenizer_json"].ElementCount));

        using CudaBackend backend = new CudaBackend(0, PtxDir());
        _output.WriteLine($"device: {backend.Context.DeviceName}");
        int[] firstIds = tokenizer.Encode("a samurai on a neon rooftop at night");
        int[] secondIds = tokenizer.Encode("a bowl of fruit on a wooden table");
        int[] first = Gemma4Tokenizer.BuildConditioningSequence(firstIds);
        int[] second = Gemma4Tokenizer.BuildConditioningSequence(secondIds);
        Assert.Equal(Gemma4Tokenizer.LtxMinLength, first.Length);
        int contentLength = Math.Max(firstIds.Length, secondIds.Length) + 1;   // +1 for the prepended BOS
        _output.WriteLine($"real tokens: {firstIds.Length} / {secondIds.Length} of {first.Length} "
            + $"(the rest is identical right-padding)");

        using Tensor firstStates = encoder.Encode(backend, [first]);
        using Tensor secondStates = encoder.Encode(backend, [second]);
        // Positive control: the same prompt twice. Without it a small cross-prompt difference is unreadable —
        // it could mean a nearly-dead encoder OR simply that Gemma's semantic signal rides on a large
        // prompt-independent residual. This separates those two readings.
        using Tensor firstAgain = encoder.Encode(backend, [first]);

        double firstRms = RootMeanSquare(firstStates, out long firstNonFinite);
        double secondRms = RootMeanSquare(secondStates, out long secondNonFinite);
        int hidden = (int)firstStates.Shape[2];
        double contentDifference = RelativeDifference(firstStates, secondStates, 0, contentLength * hidden);
        double paddingDifference = RelativeDifference(firstStates, secondStates,
            contentLength * hidden, (int)firstStates.ElementCount - contentLength * hidden);
        _output.WriteLine($"states {firstStates.Shape}: rms {firstRms:F4} / {secondRms:F4}, "
            + $"non-finite {firstNonFinite} / {secondNonFinite}; relative difference — "
            + $"content span {contentDifference:F4}, padded tail {paddingDifference:F4}");

        int hiddenSize = (int)firstStates.Shape[2];
        double controlDifference = RelativeDifference(firstStates, firstAgain, 0, contentLength * hiddenSize);
        _output.WriteLine($"same-prompt control over the content span: {controlDifference:E3}");

        double maxSharedTokenDifference = 0, maxDifferingTokenDifference = 0;
        for (int position = 0; position < contentLength; position++)
        {
            double atPosition = RelativeDifference(firstStates, secondStates, (long)position * hiddenSize, hiddenSize);
            bool sameToken = first[position] == second[position];
            _output.WriteLine($"  position {position,2} (token {first[position],6} vs {second[position],6}, "
                + $"{(sameToken ? "same" : "differ")}): relative difference {atPosition:F4}");
            if (sameToken) maxSharedTokenDifference = Math.Max(maxSharedTokenDifference, atPosition);
            else maxDifferingTokenDifference = Math.Max(maxDifferingTokenDifference, atPosition);
        }

        Assert.Equal(0, firstNonFinite);
        Assert.Equal(0, secondNonFinite);

        // A dropped rotation or a mis-broadcast row scale compounds across 48 blocks: the residual either
        // saturates or decays to nothing. Neither can land inside a band this narrow by accident.
        Assert.InRange(firstRms, 0.05, 500.0);
        Assert.InRange(secondRms, 0.05, 500.0);

        // Determinism first: the resident int8 path picks its GEMM row chunk from live free VRAM, so if that
        // choice could perturb the result at all it would show up here. int32 accumulation is exact and
        // order-independent, so it cannot — measured exactly 0.
        Assert.Equal(0.0, controlDifference);

        // The real gate is STRUCTURAL, not a magnitude: with a causal mask, positions whose token AND whose
        // whole prefix match must come out bit-identical, and positions after a differing token must not.
        // Magnitude alone is a bad gate here — Gemma's residual carries a large prompt-independent component
        // (rms ~45), so genuine semantic divergence reads as only a few percent.
        Assert.Equal(0.0, maxSharedTokenDifference);
        Assert.True(maxDifferingTokenDifference > 0.01,
            $"positions with differing tokens produced near-identical states (max {maxDifferingTokenDifference:E3}) — "
            + "the tower is not discriminating between prompts.");
    }

    private static unsafe double RootMeanSquare(Tensor states, out long nonFinite)
    {
        using Tensor f32 = states.DType == DType.F32 ? states.Reshape(states.Shape) : states.CastTo(DType.F32);
        float* values = (float*)f32.DataPointer;
        long count = f32.ElementCount;
        double sumSquares = 0;
        nonFinite = 0;
        for (long i = 0; i < count; i++)
        {
            float value = values[i];
            if (!float.IsFinite(value)) { nonFinite++; continue; }
            sumSquares += (double)value * value;
        }
        return Math.Sqrt(sumSquares / count);
    }

    /// <summary>relL2 between two state tensors over a flat element window.</summary>
    private static unsafe double RelativeDifference(Tensor a, Tensor b, long offset, long count)
    {
        using Tensor left = a.DType == DType.F32 ? a.Reshape(a.Shape) : a.CastTo(DType.F32);
        using Tensor right = b.DType == DType.F32 ? b.Reshape(b.Shape) : b.CastTo(DType.F32);
        float* p = (float*)left.DataPointer;
        float* q = (float*)right.DataPointer;
        long end = Math.Min(offset + count, Math.Min(left.ElementCount, right.ElementCount));
        double numerator = 0, denominator = 0;
        for (long i = offset; i < end; i++)
        {
            double difference = p[i] - q[i];
            numerator += difference * difference;
            denominator += (double)p[i] * p[i];
        }
        return denominator > 0 ? Math.Sqrt(numerator / denominator) : 0;
    }
}
