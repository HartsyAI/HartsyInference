using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Regression coverage for the device-resident attention glue used by <see cref="LlamaStyleEncoder"/>:
/// flat/token-major projections → head-major Q/K/V, optional per-head RMSNorm, split-half RoPE,
/// grouped-query K/V expansion, and the inverse head-major → flat layout.
/// </summary>
/// <remarks>
/// The numerical oracle below is deliberately independent scalar code. The residency assertions use
/// <see cref="CudaBackend.GetD2hSyncCount"/> so a host-pointer fallback cannot pass merely because it is
/// numerically correct. Activations in <see cref="LlamaStyleEncoder"/> are currently F32; F16/BF16 cases cover
/// the bit-preserving layout and repeat primitives so a later mixed-precision encoder does not regress them.
/// </remarks>
[Collection("CudaSerial")]
public sealed unsafe class LlamaAttentionGpuGlueTests
{
    private readonly ITestOutputHelper _output;

    public LlamaAttentionGpuGlueTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>
    /// These are the distinct attention geometries exercised by image-model presets, plus a deliberately
    /// non-GQA case. All sequence lengths are odd so a block/tile-tail indexing bug cannot hide.
    /// </summary>
    public static TheoryData<string, int, int, int, int, int, bool, float> ImageModelShapes() => new()
    {
        // Anima: Qwen3-0.6B (its Q projection is wider than hidden, but the attention geometry is 16:8).
        { "Qwen3-0.6B",     1, 16, 8, 15, 128, true,  1_000_000f },
        // Z-Image / Flux.2 Klein / Krea 2 / MageFlow / Ideogram 4 / Boogu: Qwen3 4B/8B families.
        { "Qwen3-4B/8B",    1, 32, 8, 17, 128, true,  1_000_000f },
        // HiDream / Flux.2 Dev / ERNIE-Image: Llama-3.1, Mistral-Small-3, and Ministral-3B.
        { "Llama-Mistral",  1, 32, 8, 23, 128, false,   500_000f },
        // Qwen-Image / HunyuanImage / Kandinsky 5: Qwen2.5-VL-7B.
        { "Qwen2.5-VL-7B", 1, 28, 4, 31, 128, false, 1_000_000f },
        // OmniGen2: Qwen2.5-VL-3B (the widest KV repeat factor among the image presets: group=8).
        { "Qwen2.5-VL-3B", 1, 16, 2, 21, 128, false, 1_000_000f },
        // Lumina 2: Gemma 2 has D=256 without per-head Q/K normalization.
        { "Gemma2-2B",      1,  8, 4, 25, 256, false,    10_000f },
        // LTX uses this through the same encoder; it is also the D=256 + head-norm boundary case.
        { "Gemma3-12B",     1, 16, 8, 19, 256, true,  1_000_000f },
        // No current image preset is MHA, but group=1 is a required contract for future presets/refactors.
        { "Synthetic-MHA",  2,  7, 7, 13,  64, false,    10_000f },
    };

    [Theory]
    [MemberData(nameof(ImageModelShapes))]
    [Trait("Category", "GpuIntegration")]
    public void CudaF32_AttentionGlue_MatchesIndependentReference_WithoutIntermediateD2h(
        string model, int batch, int queryHeads, int kvHeads, int sequence, int headDim,
        bool headNorm, float ropeTheta)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        int queryWidth = queryHeads * headDim;
        int kvWidth = kvHeads * headDim;
        float[] qFlatValues = RandomValues(batch * sequence * queryWidth, seed: 101 + queryHeads);
        float[] kFlatValues = RandomValues(batch * sequence * kvWidth, seed: 211 + kvHeads);
        float[] vFlatValues = RandomValues(batch * sequence * kvWidth, seed: 307 + sequence);
        float[] qNormWeight = RandomWeights(headDim, seed: 401 + headDim);
        float[] kNormWeight = RandomWeights(headDim, seed: 503 + headDim);
        (float[] ropeCosHalf, float[] ropeSinHalf) = BuildHalfRope(sequence, headDim, ropeTheta);

        GlueReference expected = ReferenceGlue(
            qFlatValues, kFlatValues, vFlatValues, qNormWeight, kNormWeight,
            batch, queryHeads, kvHeads, sequence, headDim, headNorm, ropeCosHalf, ropeSinHalf);

        using Tensor qFlat = TensorFrom(qFlatValues, new TensorShape(batch, sequence, queryWidth));
        using Tensor kFlat = TensorFrom(kFlatValues, new TensorShape(batch, sequence, kvWidth));
        using Tensor vFlat = TensorFrom(vFlatValues, new TensorShape(batch, sequence, kvWidth));
        using Tensor qWeight = TensorFrom(qNormWeight, new TensorShape(headDim));
        using Tensor kWeight = TensorFrom(kNormWeight, new TensorShape(headDim));
        using Tensor cos = FullWidthRopeTensor(ropeCosHalf, batch, sequence, headDim);
        using Tensor sin = FullWidthRopeTensor(ropeSinHalf, batch, sequence, headDim);
        using Tensor qHeadsRaw = new(new TensorShape(batch, queryHeads, sequence, headDim), DType.F32);
        using Tensor kHeadsRaw = new(new TensorShape(batch, kvHeads, sequence, headDim), DType.F32);
        using Tensor vHeads = new(new TensorShape(batch, kvHeads, sequence, headDim), DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        IBackend backend = cuda;

        cuda.ResetD2hSyncCount();
        backend.Permute0213(qHeadsRaw, qFlat, sequence, queryHeads, headDim);
        backend.Permute0213(kHeadsRaw, kFlat, sequence, kvHeads, headDim);
        backend.Permute0213(vHeads, vFlat, sequence, kvHeads, headDim);

        Tensor qHeads = qHeadsRaw;
        Tensor kHeads = kHeadsRaw;
        Tensor? qNormalized = null;
        Tensor? kNormalized = null;
        Tensor? kRepeated = null;
        Tensor? vRepeated = null;
        try
        {
            if (headNorm)
            {
                qNormalized = new Tensor(qHeadsRaw.Shape, DType.F32);
                kNormalized = new Tensor(kHeadsRaw.Shape, DType.F32);
                backend.RmsNorm(qNormalized, qHeadsRaw, qWeight, 1e-6f);
                backend.RmsNorm(kNormalized, kHeadsRaw, kWeight, 1e-6f);
                qHeads = qNormalized;
                kHeads = kNormalized;
            }

            backend.ApplyRopeSingleHeadMajor(qHeads, cos, sin);
            backend.ApplyRopeSingleHeadMajor(kHeads, cos, sin);

            Tensor expandedK = kHeads;
            Tensor expandedV = vHeads;
            if (kvHeads != queryHeads)
            {
                kRepeated = new Tensor(new TensorShape(batch, queryHeads, sequence, headDim), DType.F32);
                vRepeated = new Tensor(new TensorShape(batch, queryHeads, sequence, headDim), DType.F32);
                backend.RepeatKvHeads(kRepeated, kHeads, kvHeads, queryHeads / kvHeads);
                backend.RepeatKvHeads(vRepeated, vHeads, kvHeads, queryHeads / kvHeads);
                expandedK = kRepeated;
                expandedV = vRepeated;
            }

            using Tensor mergedQuery = new(new TensorShape(batch, sequence, queryWidth), DType.F32);
            backend.Permute0213(mergedQuery, qHeads, queryHeads, sequence, headDim);
            cuda.Sync();

            // The entire prologue/epilogue chain ran after the reset. Any CPU fallback in permutation, norm,
            // RoPE, repeat, or inverse permutation would have invoked a lazy activation readback here.
            Assert.Equal(0, cuda.GetD2hSyncCount());

            AssertClose(expected.MergedQuery, mergedQuery, 8e-5f, $"{model} merged Q");
            AssertClose(expected.RepeatedKey, expandedK, 8e-5f, $"{model} repeated K");
            AssertClose(expected.RepeatedValue, expandedV, 0f, $"{model} repeated V");
            Assert.Equal(3, cuda.GetD2hSyncCount());
        }
        finally
        {
            kRepeated?.Dispose();
            vRepeated?.Dispose();
            qNormalized?.Dispose();
            kNormalized?.Dispose();
        }
    }

    [Theory]
    [InlineData("F32", 2, 7, 7, 13, 64)]
    [InlineData("F16", 1, 28, 4, 17, 128)]
    [InlineData("BF16", 1, 32, 8, 19, 128)]
    [Trait("Category", "GpuIntegration")]
    public void CudaLayoutRepeatMerge_AllActivationDtypes_AreBitExactAndDeviceResident(
        string dtypeName, int batch, int queryHeads, int kvHeads, int sequence, int headDim)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        DType dtype = dtypeName switch
        {
            "F32" => DType.F32,
            "F16" => DType.F16,
            "BF16" => DType.BF16,
            _ => throw new ArgumentOutOfRangeException(nameof(dtypeName)),
        };
        int elementBytes = dtype.SizeInBytes;
        int inputElements = checked(batch * sequence * kvHeads * headDim);
        byte[] source = new byte[checked(inputElements * elementBytes)];
        new Random(701 + queryHeads + elementBytes).NextBytes(source);
        byte[] expected = RepeatFlatReferenceBytes(
            source, elementBytes, batch, queryHeads, kvHeads, sequence, headDim);

        using Tensor flat = new(new TensorShape(batch, sequence, kvHeads, headDim), dtype);
        fixed (byte* sourcePointer = source)
            Buffer.MemoryCopy(sourcePointer, (void*)flat.DataPointer, source.Length, source.Length);
        using Tensor headMajor = new(new TensorShape(batch, kvHeads, sequence, headDim), dtype);
        using Tensor repeated = new(new TensorShape(batch, queryHeads, sequence, headDim), dtype);
        using Tensor merged = new(new TensorShape(batch, sequence, queryHeads, headDim), dtype);
        using CudaBackend cuda = new(0, PtxDir());
        IBackend backend = cuda;

        cuda.ResetD2hSyncCount();
        backend.Permute0213(headMajor, flat, sequence, kvHeads, headDim);
        backend.RepeatKvHeads(repeated, headMajor, kvHeads, queryHeads / kvHeads);
        backend.Permute0213(merged, repeated, queryHeads, sequence, headDim);
        cuda.Sync();

        Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertBytes(expected, merged, $"{dtypeName} B{batch} H{queryHeads}:{kvHeads} S{sequence} D{headDim}");
        Assert.Equal(1, cuda.GetD2hSyncCount());
    }

    /// <summary>
    /// Production-level guard: a complete decoder block must not invoke any activation's lazy host-sync callback.
    /// Reading the returned final hidden state is the sole expected D2H.
    /// </summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void LlamaStyleEncoder_CudaForward_HasNoIntermediateD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 2);
        int[][] tokens = [TinyTokens()];

        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using CpuBackend cpu = new();
            using Tensor expected = encoder.Encode(cpu, tokens);
            using CudaBackend cuda = new(0, PtxDir());
            cuda.HighPrecisionGemm = true;
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor hidden = encoder.Encode(cuda, tokens);
            cuda.Sync();

            long intermediateReadbacks = cuda.GetD2hSyncCount();
            _output.WriteLine($"full two-layer encoder: intermediate D2H syncs={intermediateReadbacks}");
            Assert.Equal(0, intermediateReadbacks);

            // cuBLAS is forced to full F32 here; the remaining ~1e-3 gap is the expected difference between
            // the CUDA online-softmax reduction and the scalar CPU attention accumulation order.
            AssertTensorClose(expected, hidden, absoluteTolerance: 1.5e-3f, relativeTolerance: 1e-3f,
                "full two-layer CPU/CUDA parity");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "GpuIntegration")]
    public void EncodeMultiLayer_SingleTap_MatchesCpuAndStaysDeviceResident(bool interleaved)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 4);
        int[][] tokens = TinyTokenBatch();
        int[] taps = [3];
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using CpuBackend cpu = new();
            // K=1 is layout-invariant by contract. Always use the plain single-tap result as the oracle so the
            // interleaved=true case proves it does not accidentally transpose the hidden dimension.
            using Tensor expected = encoder.EncodeMultiLayer(cpu, tokens, taps, interleavedLayout: false);
            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor actual = encoder.EncodeMultiLayer(cuda, tokens, taps, interleaved);
            cuda.Sync();

            Assert.Equal(new TensorShape(tokens.Length, tokens[0].Length, config.HiddenSize), actual.Shape);
            Assert.Equal(0, cuda.GetD2hSyncCount());
            AssertTensorClose(expected, actual, 5e-4f, 5e-4f,
                $"EncodeMultiLayer K=1 interleaved={interleaved}");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(11.313708f)] // sqrt(128): exercises the Gemma embedding-normalizer path.
    [Trait("Category", "GpuIntegration")]
    public void EncodeMultiLayer_EmbeddingTap_MatchesHostLookupAndStaysDeviceResident(float embeddingScale)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 2, embeddingScale);
        int[] tokens = TinyTokens();
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using Tensor expected = encoder.LookupEmbeddings(tokens);
            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor actual = encoder.EncodeMultiLayer(cuda, [tokens], [0]);
            cuda.Sync();

            Assert.Equal(0, cuda.GetD2hSyncCount());
            AssertTensorClose(expected, actual, 0f, 0f, "CUDA GatherRows embedding lookup");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EncodeMultiLayer_UncachedEmbeddingTable_UsesHostGatherWithoutUploadingWholeVocabulary()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 1);
        int[] tokens = TinyTokens();
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using Tensor expected = encoder.LookupEmbeddings(tokens);
            using CudaBackend cuda = new(0, PtxDir());

            using Tensor actual = encoder.EncodeMultiLayer(cuda, [tokens], [0]);

            Assert.False(GpuTransferHelper.IsWeightCached(weights["model.embed_tokens.weight"]));
            AssertTensorClose(expected, actual, 0f, 0f, "uncached vocabulary host-gather fallback");
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "GpuIntegration")]
    public void EncodeMultiLayer_ThreeTaps_HasExactRequestedMappingAndNoIntermediateD2h(bool interleaved)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 4);
        int[][] tokens = TinyTokenBatch();
        int[] taps = [0, 2, 4];
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using CpuBackend cpu = new();

            // Build the oracle from independent K=1 captures and pack it explicitly. This does not share
            // EncodeMultiLayer's Concat/Transpose implementation, so swapping K/H or tap order is observable.
            float[][] singleTapValues = new float[taps.Length][];
            for (int k = 0; k < taps.Length; k++)
            {
                using Tensor single = encoder.EncodeMultiLayer(cpu, tokens, [taps[k]]);
                singleTapValues[k] = CopyF32(single);
            }
            float[] expected = PackLayerTaps(
                singleTapValues, tokens.Length, tokens[0].Length, config.HiddenSize, interleaved);

            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor actual = encoder.EncodeMultiLayer(cuda, tokens, taps, interleaved);
            cuda.Sync();

            Assert.Equal(
                new TensorShape(tokens.Length, tokens[0].Length, taps.Length * config.HiddenSize), actual.Shape);
            Assert.Equal(0, cuda.GetD2hSyncCount());
            AssertClose(expected, actual, 5e-4f,
                $"EncodeMultiLayer K=3 interleaved={interleaved} exact channel map");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EncodeEmbedsMrope_TextOnly_MatchesStandardCpuEncodeAndStaysDeviceResident()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 3);
        int[] tokens = TinyTokens();
        (float[] cos, float[] sin) = BuildHalfRope(tokens.Length, config.HeadDim, config.RopeTheta);
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using Tensor embeds = encoder.LookupEmbeddings(tokens);
            using CpuBackend cpu = new();
            using Tensor expected = encoder.Encode(cpu, [tokens]);
            using Tensor cpuMrope = encoder.EncodeEmbedsMrope(cpu, embeds, cos, sin);
            AssertTensorClose(expected, cpuMrope, 1e-6f, 1e-6f,
                "CPU standard encode vs text-only M-RoPE");

            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor actual = encoder.EncodeEmbedsMrope(cuda, embeds, cos, sin);
            cuda.Sync();

            Assert.Equal(0, cuda.GetD2hSyncCount());
            AssertTensorClose(expected, actual, 1.2e-3f, 1e-3f, "text-only M-RoPE CPU/CUDA parity");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EncodeEmbedsMrope_Deepstack_MatchesCpuAndStaysDeviceResident()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 3);
        int[] tokens = TinyTokens();
        bool[] visualMask = Enumerable.Range(0, tokens.Length).Select(i => i is 2 or 7 or 13).ToArray();
        int visualRows = visualMask.Count(value => value);
        (float[] cos, float[] sin) = BuildHalfRope(tokens.Length, config.HeadDim, config.RopeTheta);
        Dictionary<string, Tensor> weights = BuildTinyWeights(config);
        Tensor[] deepstack =
        [
            TensorFrom(RandomScaledValues(visualRows * config.HiddenSize, 1201, 0.025f),
                new TensorShape(visualRows, config.HiddenSize)),
            TensorFrom(RandomScaledValues(visualRows * config.HiddenSize, 1202, 0.04f),
                new TensorShape(visualRows, config.HiddenSize)),
        ];
        try
        {
            using LlamaStyleEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using Tensor embeds = encoder.LookupEmbeddings(tokens);
            using CpuBackend cpu = new();
            using Tensor expected = encoder.EncodeEmbedsMrope(cpu, embeds, cos, sin, deepstack, visualMask);
            using Tensor withoutDeepstack = encoder.EncodeEmbedsMrope(cpu, embeds, cos, sin);
            AssertMateriallyDifferent(expected, withoutDeepstack, "deepstack injection must affect the hidden state");

            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            cuda.PreloadWeights(encoder.EnumerateWeights());
            cuda.ResetD2hSyncCount();

            using Tensor actual = encoder.EncodeEmbedsMrope(cuda, embeds, cos, sin, deepstack, visualMask);
            cuda.Sync();

            Assert.Equal(0, cuda.GetD2hSyncCount());
            AssertTensorClose(expected, actual, 7e-4f, 7e-4f, "deepstack M-RoPE CPU/CUDA parity");
            Assert.Equal(1, cuda.GetD2hSyncCount());
        }
        finally
        {
            DisposeAll(deepstack);
            DisposeAll(weights.Values);
        }
    }

    [Fact]
    public void EncodeMultiLayer_RejectsInvalidLayerAndTokenContracts()
    {
        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 4);
        using LlamaStyleEncoder encoder = new(config);
        using CpuBackend cpuImplementation = new();
        IBackend cpu = cpuImplementation;
        int[][] tokens = [TinyTokens()];

        Assert.Throws<ArgumentException>(() => encoder.EncodeMultiLayer(cpu, tokens, []));
        Assert.Throws<ArgumentException>(() => encoder.EncodeMultiLayer(cpu, tokens, [2, 2]));
        Assert.Throws<ArgumentException>(() => encoder.EncodeMultiLayer(cpu, tokens, [3, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.EncodeMultiLayer(cpu, tokens, [5]));
        Assert.Throws<ArgumentException>(() => encoder.EncodeMultiLayer(cpu, [TinyTokens(), [1, 2]], [1]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            encoder.EncodeMultiLayer(cpu, [[1, config.VocabSize]], [1]));
    }

    [Fact]
    public void ApplyRopeSingleHeadMajor_RejectsShortSineTableBeforeReadingIt()
    {
        using Tensor x = new(new TensorShape(1, 2, 3, 4), DType.F32);
        using Tensor cos = new(new TensorShape(1, 3, 4), DType.F32);
        using Tensor shortSin = new(new TensorShape(1, 2, 4), DType.F32);
        using CpuBackend cpuImplementation = new();
        IBackend cpu = cpuImplementation;

        Assert.Throws<ArgumentException>(() => cpu.ApplyRopeSingleHeadMajor(x, cos, shortSin));

        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED CUDA assertion: CUDA unavailable");
            return;
        }
        using CudaBackend cuda = new(0, PtxDir());
        Assert.Throws<HartsyInferenceException>(() => cuda.ApplyRopeSingleHeadMajor(x, cos, shortSin));
    }

    [Fact]
    public void GatherRows_RejectsInvalidGeometryIndicesAndAliasing()
    {
        using Tensor input = new(new TensorShape(3, 4), DType.F32);
        using Tensor output = new(new TensorShape(2, 4), DType.F32);
        using Tensor shortOutput = new(new TensorShape(1, 4), DType.F32);
        using CpuBackend cpuImplementation = new();
        IBackend cpu = cpuImplementation;

        Assert.Throws<ArgumentException>(() => cpu.GatherRows(shortOutput, input, [0, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.GatherRows(output, input, [-1, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.GatherRows(output, input, [0, 3]));
        Assert.Throws<ArgumentException>(() => cpu.GatherRows(input, input, [0, 1, 2]));

        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED CUDA assertions: CUDA unavailable");
            return;
        }
        using CudaBackend cuda = new(0, PtxDir());
        Assert.Throws<ArgumentException>(() => cuda.GatherRows(shortOutput, input, [0, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.GatherRows(output, input, [-1, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.GatherRows(output, input, [0, 3]));
        Assert.Throws<ArgumentException>(() => cuda.GatherRows(input, input, [0, 1, 2]));
    }

    [Fact]
    public void EncodeEmbedsMrope_RejectsInvalidInputRopeAndDeepstackContracts()
    {
        LlamaStyleEncoderConfig config = TinyConfig(numLayers: 2);
        using LlamaStyleEncoder encoder = new(config);
        using CpuBackend cpu = new();
        const int Sequence = 5;
        (float[] cos, float[] sin) = BuildHalfRope(Sequence, config.HeadDim, config.RopeTheta);
        using Tensor valid = new(new TensorShape(1, Sequence, config.HiddenSize), DType.F32);
        using Tensor wrongDtype = new(valid.Shape, DType.F16);
        using Tensor wrongHidden = new(new TensorShape(1, Sequence, config.HiddenSize - 1), DType.F32);
        using Tensor feature = new(new TensorShape(2, config.HiddenSize), DType.F32);
        using Tensor wrongFeature = new(new TensorShape(1, config.HiddenSize), DType.F32);
        bool[] mask = [false, true, false, true, false];

        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, wrongDtype, cos, sin));
        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, wrongHidden, cos, sin));
        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, valid, [1f], [0f]));
        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, valid, cos, sin, [feature], null));
        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, valid, cos, sin, [feature], new bool[Sequence - 1]));
        Assert.Throws<ArgumentException>(() => encoder.EncodeEmbedsMrope(cpu, valid, cos, sin, [wrongFeature], mask));
        Assert.Throws<ArgumentException>(() =>
            encoder.EncodeEmbedsMrope(cpu, valid, cos, sin, [feature], new bool[Sequence]));
    }

    private sealed record GlueReference(float[] MergedQuery, float[] RepeatedKey, float[] RepeatedValue);

    private static GlueReference ReferenceGlue(
        float[] qFlat, float[] kFlat, float[] vFlat, float[] qWeight, float[] kWeight,
        int batch, int queryHeads, int kvHeads, int sequence, int headDim, bool headNorm,
        float[] ropeCosHalf, float[] ropeSinHalf)
    {
        float[] q = FlatToHeadsReference(qFlat, batch, sequence, queryHeads, headDim);
        float[] k = FlatToHeadsReference(kFlat, batch, sequence, kvHeads, headDim);
        float[] v = FlatToHeadsReference(vFlat, batch, sequence, kvHeads, headDim);
        if (headNorm)
        {
            RmsNormRowsInPlace(q, headDim, qWeight, 1e-6f);
            RmsNormRowsInPlace(k, headDim, kWeight, 1e-6f);
        }
        RopeHeadMajorInPlace(q, batch, queryHeads, sequence, headDim, ropeCosHalf, ropeSinHalf);
        RopeHeadMajorInPlace(k, batch, kvHeads, sequence, headDim, ropeCosHalf, ropeSinHalf);

        int group = queryHeads / kvHeads;
        float[] repeatedK = RepeatHeadsReference(k, batch, kvHeads, group, sequence, headDim);
        float[] repeatedV = RepeatHeadsReference(v, batch, kvHeads, group, sequence, headDim);
        float[] mergedQ = HeadsToFlatReference(q, batch, queryHeads, sequence, headDim);
        return new GlueReference(mergedQ, repeatedK, repeatedV);
    }

    private static float[] FlatToHeadsReference(float[] input, int batch, int sequence, int heads, int headDim)
    {
        float[] output = new float[input.Length];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < sequence; s++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < headDim; d++)
                        output[(((b * heads) + h) * sequence + s) * headDim + d] =
                            input[(((b * sequence) + s) * heads + h) * headDim + d];
        return output;
    }

    private static float[] HeadsToFlatReference(float[] input, int batch, int heads, int sequence, int headDim)
    {
        float[] output = new float[input.Length];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < sequence; s++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < headDim; d++)
                        output[(((b * sequence) + s) * heads + h) * headDim + d] =
                            input[(((b * heads) + h) * sequence + s) * headDim + d];
        return output;
    }

    private static float[] RepeatHeadsReference(
        float[] input, int batch, int kvHeads, int group, int sequence, int headDim)
    {
        int queryHeads = kvHeads * group;
        float[] output = new float[batch * queryHeads * sequence * headDim];
        for (int b = 0; b < batch; b++)
            for (int qh = 0; qh < queryHeads; qh++)
                for (int s = 0; s < sequence; s++)
                    for (int d = 0; d < headDim; d++)
                        output[(((b * queryHeads) + qh) * sequence + s) * headDim + d] =
                            input[(((b * kvHeads) + qh / group) * sequence + s) * headDim + d];
        return output;
    }

    private static void RmsNormRowsInPlace(float[] values, int rowWidth, float[] weight, float eps)
    {
        for (int row = 0; row < values.Length / rowWidth; row++)
        {
            int offset = row * rowWidth;
            double sumSquares = 0.0;
            for (int d = 0; d < rowWidth; d++)
                sumSquares += (double)values[offset + d] * values[offset + d];
            float inverseRms = (float)(1.0 / Math.Sqrt(sumSquares / rowWidth + eps));
            for (int d = 0; d < rowWidth; d++)
                values[offset + d] *= inverseRms * weight[d];
        }
    }

    private static void RopeHeadMajorInPlace(
        float[] values, int batch, int heads, int sequence, int headDim,
        float[] cosHalf, float[] sinHalf)
    {
        int half = headDim / 2;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < sequence; s++)
                    for (int i = 0; i < half; i++)
                    {
                        int offset = (((b * heads) + h) * sequence + s) * headDim;
                        float lower = values[offset + i];
                        float upper = values[offset + half + i];
                        float c = cosHalf[s * half + i];
                        float si = sinHalf[s * half + i];
                        values[offset + i] = lower * c - upper * si;
                        values[offset + half + i] = upper * c + lower * si;
                    }
    }

    private static (float[] Cos, float[] Sin) BuildHalfRope(int sequence, int headDim, float theta)
    {
        int half = headDim / 2;
        float[] cos = new float[sequence * half];
        float[] sin = new float[sequence * half];
        for (int s = 0; s < sequence; s++)
            for (int i = 0; i < half; i++)
            {
                double frequency = 1.0 / Math.Pow(theta, (double)(2 * i) / headDim);
                double angle = s * frequency;
                cos[s * half + i] = (float)Math.Cos(angle);
                sin[s * half + i] = (float)Math.Sin(angle);
            }
        return (cos, sin);
    }

    private static Tensor FullWidthRopeTensor(float[] halfValues, int batch, int sequence, int headDim)
    {
        Tensor tensor = new(new TensorShape(batch, sequence, headDim), DType.F32);
        float* output = (float*)tensor.DataPointer;
        int half = headDim / 2;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < sequence; s++)
                for (int i = 0; i < half; i++)
                {
                    float value = halfValues[s * half + i];
                    long row = ((long)b * sequence + s) * headDim;
                    output[row + i] = value;
                    output[row + half + i] = value;
                }
        return tensor;
    }

    private static float[] RandomValues(int count, int seed)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static float[] RandomWeights(int count, int seed)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = 0.75f + (float)random.NextDouble() * 0.5f;
        return values;
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        fixed (float* source = values)
        {
            long bytes = values.LongLength * sizeof(float);
            Buffer.MemoryCopy(source, (void*)tensor.DataPointer, bytes, bytes);
        }
        return tensor;
    }

    private void AssertClose(float[] expected, Tensor actual, float tolerance, string label)
    {
        Assert.Equal(expected.LongLength, actual.ElementCount);
        float* values = (float*)actual.DataPointer;
        float maxError = 0f;
        long maxIndex = 0;
        for (long i = 0; i < actual.ElementCount; i++)
        {
            float error = MathF.Abs(expected[i] - values[i]);
            if (error > maxError)
            {
                maxError = error;
                maxIndex = i;
            }
        }
        _output.WriteLine($"{label}: max abs error={maxError:E3} at {maxIndex}");
        Assert.True(maxError <= tolerance,
            $"{label}: max abs error {maxError:E3} exceeds {tolerance:E3} at {maxIndex}.");
    }

    private static byte[] RepeatFlatReferenceBytes(
        byte[] source, int elementBytes, int batch, int queryHeads, int kvHeads, int sequence, int headDim)
    {
        int group = queryHeads / kvHeads;
        byte[] output = new byte[checked(batch * sequence * queryHeads * headDim * elementBytes)];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < sequence; s++)
                for (int qh = 0; qh < queryHeads; qh++)
                    for (int d = 0; d < headDim; d++)
                    {
                        int inputElement = (((b * sequence) + s) * kvHeads + qh / group) * headDim + d;
                        int outputElement = (((b * sequence) + s) * queryHeads + qh) * headDim + d;
                        Buffer.BlockCopy(source, inputElement * elementBytes, output, outputElement * elementBytes, elementBytes);
                    }
        return output;
    }

    private void AssertBytes(byte[] expected, Tensor actual, string label)
    {
        int byteCount = checked((int)(actual.ElementCount * actual.DType.SizeInBytes));
        Assert.Equal(expected.Length, byteCount);
        byte* actualPointer = (byte*)actual.DataPointer;
        int firstBad = -1;
        for (int i = 0; i < byteCount; i++)
        {
            if (expected[i] == actualPointer[i]) continue;
            firstBad = i;
            break;
        }
        _output.WriteLine($"{label}: {byteCount} bytes checked bit-exact");
        if (firstBad >= 0)
            Assert.Fail(
                $"{label}: first mismatch at byte {firstBad}: expected 0x{expected[firstBad]:X2}, actual 0x{actualPointer[firstBad]:X2}.");
    }

    private void AssertTensorClose(
        Tensor expected, Tensor actual, float absoluteTolerance, float relativeTolerance, string label)
    {
        Assert.Equal(DType.F32, expected.DType);
        Assert.Equal(DType.F32, actual.DType);
        Assert.Equal(expected.Shape, actual.Shape);
        float* expectedValues = (float*)expected.DataPointer;
        float* actualValues = (float*)actual.DataPointer;
        float maxAbsoluteError = 0f;
        float maxRelativeError = 0f;
        long maxIndex = 0;
        long violations = 0;
        long firstViolation = -1;
        float firstExpected = 0f;
        float firstActual = 0f;
        for (long i = 0; i < actual.ElementCount; i++)
        {
            float expectedValue = expectedValues[i];
            float actualValue = actualValues[i];
            Assert.True(float.IsFinite(expectedValue), $"{label}: reference is non-finite at {i}: {expectedValue}.");
            Assert.True(float.IsFinite(actualValue), $"{label}: CUDA result is non-finite at {i}: {actualValue}.");
            float absoluteError = MathF.Abs(expectedValue - actualValue);
            float relativeError = absoluteError / MathF.Max(MathF.Abs(expectedValue), 1e-6f);
            if (absoluteError > maxAbsoluteError)
            {
                maxAbsoluteError = absoluteError;
                maxRelativeError = relativeError;
                maxIndex = i;
            }
            if (absoluteError > absoluteTolerance + relativeTolerance * MathF.Abs(expectedValue))
            {
                violations++;
                if (firstViolation < 0)
                {
                    firstViolation = i;
                    firstExpected = expectedValue;
                    firstActual = actualValue;
                }
            }
        }
        _output.WriteLine(
            $"{label}: max abs error={maxAbsoluteError:E3}, relative={maxRelativeError:E3} at {maxIndex}");
        Assert.True(violations == 0,
            $"{label}: {violations} values exceeded tolerance; first at {firstViolation}: " +
            $"expected={firstExpected:G9}, actual={firstActual:G9}.");
    }

    private static float[] CopyF32(Tensor tensor)
    {
        if (tensor.DType != DType.F32)
            throw new ArgumentException($"Expected F32 tensor, got {tensor.DType}.", nameof(tensor));
        float[] values = new float[checked((int)tensor.ElementCount)];
        float* source = (float*)tensor.DataPointer;
        for (int i = 0; i < values.Length; i++) values[i] = source[i];
        return values;
    }

    private static float[] PackLayerTaps(
        IReadOnlyList<float[]> taps, int batch, int sequence, int hidden, bool interleaved)
    {
        int count = taps.Count;
        int valuesPerTap = checked(batch * sequence * hidden);
        foreach (float[] tap in taps)
            if (tap.Length != valuesPerTap)
                throw new ArgumentException($"Tap has {tap.Length} values; expected {valuesPerTap}.", nameof(taps));

        float[] output = new float[checked(valuesPerTap * count)];
        for (int token = 0; token < batch * sequence; token++)
            for (int k = 0; k < count; k++)
                for (int h = 0; h < hidden; h++)
                {
                    int destination = interleaved
                        ? token * hidden * count + h * count + k
                        : token * hidden * count + k * hidden + h;
                    output[destination] = taps[k][token * hidden + h];
                }
        return output;
    }

    private static void AssertMateriallyDifferent(Tensor first, Tensor second, string message)
    {
        Assert.Equal(first.Shape, second.Shape);
        float* a = (float*)first.DataPointer;
        float* b = (float*)second.DataPointer;
        float maxDifference = 0f;
        for (long i = 0; i < first.ElementCount; i++)
            maxDifference = MathF.Max(maxDifference, MathF.Abs(a[i] - b[i]));
        Assert.True(maxDifference > 1e-4f, $"{message}; max difference was only {maxDifference:E3}.");
    }

    private static float[] RandomScaledValues(int count, int seed, float scale)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)((random.NextDouble() * 2.0 - 1.0) * scale);
        return values;
    }

    private static int[] TinyTokens() =>
        [1, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 2, 3, 5, 13];

    private static int[][] TinyTokenBatch() =>
    [
        TinyTokens(),
        [13, 5, 3, 2, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19, 11, 7, 1],
    ];

    private static LlamaStyleEncoderConfig TinyConfig(int numLayers, float embeddingScale = 1.0f) => new()
    {
        HiddenSize = 128,
        NumLayers = numLayers,
        NumQueryHeads = 4,
        NumKvHeads = 2,
        HeadDim = 32,
        IntermediateSize = 192,
        VocabSize = 64,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 64,
        QkHeadNorm = true,
        AttentionBias = false,
        HasFinalNorm = true,
        EmbeddingScale = embeddingScale,
        EosTokenId = 1,
        BosTokenId = 0,
    };

    private static void DisposeAll(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors) tensor.Dispose();
    }

    private static Dictionary<string, Tensor> BuildTinyWeights(LlamaStyleEncoderConfig config)
    {
        Dictionary<string, Tensor> weights = new();
        int seed = 900;
        int hidden = config.HiddenSize;
        int queryWidth = config.NumQueryHeads * config.HeadDim;
        int kvWidth = config.NumKvHeads * config.HeadDim;

        Tensor Matrix(string key, int rows, int columns, float scale)
        {
            Tensor tensor = new(new TensorShape(rows, columns), DType.F32);
            float* values = (float*)tensor.DataPointer;
            Random random = new(seed++);
            for (long i = 0; i < tensor.ElementCount; i++)
                values[i] = (float)((random.NextDouble() * 2.0 - 1.0) * scale);
            weights[key] = tensor;
            return tensor;
        }

        Tensor Vector(string key, int count)
        {
            Tensor tensor = new(new TensorShape(count), DType.F32);
            float* values = (float*)tensor.DataPointer;
            Random random = new(seed++);
            for (long i = 0; i < tensor.ElementCount; i++)
                values[i] = 0.9f + (float)random.NextDouble() * 0.2f;
            weights[key] = tensor;
            return tensor;
        }

        Matrix("model.embed_tokens.weight", config.VocabSize, hidden, 0.05f);
        Vector("model.norm.weight", hidden);
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            string prefix = $"model.layers.{layer}";
            Vector($"{prefix}.input_layernorm.weight", hidden);
            Vector($"{prefix}.post_attention_layernorm.weight", hidden);
            Matrix($"{prefix}.self_attn.q_proj.weight", queryWidth, hidden, 0.04f);
            Matrix($"{prefix}.self_attn.k_proj.weight", kvWidth, hidden, 0.04f);
            Matrix($"{prefix}.self_attn.v_proj.weight", kvWidth, hidden, 0.04f);
            Matrix($"{prefix}.self_attn.o_proj.weight", hidden, queryWidth, 0.04f);
            Vector($"{prefix}.self_attn.q_norm.weight", config.HeadDim);
            Vector($"{prefix}.self_attn.k_norm.weight", config.HeadDim);
            Matrix($"{prefix}.mlp.gate_proj.weight", config.IntermediateSize, hidden, 0.03f);
            Matrix($"{prefix}.mlp.up_proj.weight", config.IntermediateSize, hidden, 0.03f);
            Matrix($"{prefix}.mlp.down_proj.weight", hidden, config.IntermediateSize, 0.03f);
        }
        return weights;
    }
}
