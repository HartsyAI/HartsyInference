using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for Z-Image-Base's packed two-pass CFG loop and its two-caption refiner cache.</summary>
[Collection("CudaSerial")]
public sealed unsafe class ZImagePackedCfgResidencyTests
{
    private readonly ITestOutputHelper _output;

    public ZImagePackedCfgResidencyTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void PackedCfgMapping_MatchesLegacyCondAnchoredNegatedEuler()
    {
        using CpuBackend cpu = new();
        using Tensor initial = Values([0.25f, -0.75f, 1.5f, -2.0f]);
        using Tensor actual = Values([0.25f, -0.75f, 1.5f, -2.0f]);
        using Tensor cond = Values([0.4f, -1.2f, 2.5f, 0.75f]);
        using Tensor uncond = Values([-0.2f, 0.3f, 1.0f, -1.25f]);

        const float cfg = 4.0f;
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 6.0f);
        scheduler.SetTimesteps(5);
        float dt = scheduler.Dt(2);

        float* z0 = (float*)initial.DataPointer;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        float[] expected = new float[4];
        for (int i = 0; i < expected.Length; i++)
        {
            float legacyCombined = cp[i] + cfg * (cp[i] - up[i]);
            expected[i] = z0[i] + (-legacyCombined) * dt;
        }

        // IBackend's combine is standard guidance form. g=cfg+1 maps it exactly to Z-Image's
        // cond-anchored formula; -dt folds in the model-output negation.
        ((IBackend)cpu).CfgEulerStep(actual, cond, uncond, cfg + 1.0f, -dt);
        AssertClose(expected, actual, 2e-6f);
    }

    [Theory]
    [InlineData("z_image_base-bf16.safetensors", ZImageCheckpointConverter.CheckpointVariant.Base, 6.0f)]
    [InlineData("z_image_base-nvfp8-mixed.safetensors", ZImageCheckpointConverter.CheckpointVariant.Base, 6.0f)]
    [InlineData("SwarmUI_Z-Image-Turbo-FP8Mix.safetensors", ZImageCheckpointConverter.CheckpointVariant.Turbo, 3.0f)]
    [InlineData("renamed.safetensors", ZImageCheckpointConverter.CheckpointVariant.Unknown, 3.0f)]
    [InlineData("base/Z-Image-Turbo.safetensors", ZImageCheckpointConverter.CheckpointVariant.Turbo, 3.0f)]
    [InlineData("Z-Image-base-turbo.safetensors", ZImageCheckpointConverter.CheckpointVariant.Unknown, 3.0f)]
    public void CheckpointVariant_SelectsCorrectSchedulerShift(string path,
        ZImageCheckpointConverter.CheckpointVariant expectedVariant, float expectedShift)
    {
        ZImageCheckpointConverter.CheckpointVariant variant =
            ZImageCheckpointConverter.DetectVariantFromFileName(path);
        Assert.Equal(expectedVariant, variant);
        ZImageConfig config = ZImageConfig.FromWeights(
            new Dictionary<string, Tensor>(), variant == ZImageCheckpointConverter.CheckpointVariant.Base);
        Assert.Equal(expectedShift, config.SchedulerShift);
        Assert.Equal(variant == ZImageCheckpointConverter.CheckpointVariant.Base, config.IsBase);

        GenerationDefaults diffusionDefaults = config.IsBase
            ? GenerationDefaults.ZImageBase
            : GenerationDefaults.ZImageTurbo;
        Assert.Equal(config.IsBase ? 50 : 8, diffusionDefaults.Steps);
        Assert.Equal(config.IsBase ? 5.0f : 1.0f, diffusionDefaults.CfgScale);

        if (config.IsBase)
        {
            Assert.Equal(50, ZImageRecipe.BaseDefaults.Steps);
            Assert.Equal(5.0f, ZImageRecipe.BaseDefaults.CfgScale);
            Assert.Equal(6.0, ZImageRecipe.BaseDefaults.SigmaShift);
        }
    }

    [Fact]
    public void RgbOutputGuard_RejectsOnlyExactEndpointCollapse()
    {
        InvalidOperationException black = Assert.Throws<InvalidOperationException>(
            () => ZImagePipeline.ValidateRgbOutput(new byte[12]));
        Assert.Contains("black", black.Message, StringComparison.OrdinalIgnoreCase);
        InvalidOperationException white = Assert.Throws<InvalidOperationException>(
            () => ZImagePipeline.ValidateRgbOutput(Enumerable.Repeat(byte.MaxValue, 12).ToArray()));
        Assert.Contains("white", white.Message, StringComparison.OrdinalIgnoreCase);

        ZImagePipeline.ValidateRgbOutput([0, 0, 1]);
        ZImagePipeline.ValidateRgbOutput([255, 255, 254]);
        ZImagePipeline.ValidateRgbOutput([0, 255, 0]);
    }

    [Fact]
    public void FiniteGuards_RejectNaNAndInfinityWithoutNaNComparisonFalsePositives()
    {
        using CpuBackend cpu = new();
        using Tensor finite = Values([0.0f, -2.5f, 9.0f]);
        using Tensor nan = Values([0.0f, float.NaN, 1.0f]);
        using Tensor infinity = Values([0.0f, float.PositiveInfinity, 1.0f]);

        ZImagePipeline.ValidatePredictionFinite(finite, "finite test", logStats: false);
        ZImagePipeline.ValidateFiniteTensor(cpu, finite, "finite test");
        Assert.Throws<InvalidOperationException>(
            () => ZImagePipeline.ValidatePredictionFinite(nan, "NaN test", logStats: false));
        Assert.Throws<InvalidOperationException>(
            () => ZImagePipeline.ValidatePredictionFinite(infinity, "infinity test", logStats: false));
        Assert.Throws<InvalidOperationException>(() => ZImagePipeline.ValidateFiniteTensor(cpu, nan, "NaN test"));
        Assert.Throws<InvalidOperationException>(
            () => ZImagePipeline.ValidateFiniteTensor(cpu, infinity, "infinity test"));
    }

    [Fact]
    public void BaseNumericPolicy_KeepsTokenStreamAndAttentionInF32()
    {
        ZImageConfig baseConfig = TinyConfig() with { IsBase = true, NumLayers = 1, NumRefinerLayers = 1 };
        ZImageConfig turboConfig = baseConfig with { IsBase = false };
        using ZImageTransformer baseTransformer = new(baseConfig);
        using ZImageTransformer turboTransformer = new(turboConfig);

        Assert.Equal(DType.F32, GetPrivateField<DType>(baseTransformer, "_packedActivationDtype"));
        Assert.Equal(DitDtype.Act, GetPrivateField<DType>(turboTransformer, "_packedActivationDtype"));

        object baseMainBlock = Assert.Single(GetPrivateArray(baseTransformer, "_layers"));
        object turboMainBlock = Assert.Single(GetPrivateArray(turboTransformer, "_layers"));
        object baseContextBlock = Assert.Single(GetPrivateArray(baseTransformer, "_contextRefiners"));
        object turboContextBlock = Assert.Single(GetPrivateArray(turboTransformer, "_contextRefiners"));
        Assert.False(GetPrivateField<bool>(baseMainBlock, "_allowF16Attention"));
        Assert.True(GetPrivateField<bool>(turboMainBlock, "_allowF16Attention"));
        Assert.False(GetPrivateField<bool>(baseContextBlock, "_allowF16Attention"));
        Assert.True(GetPrivateField<bool>(turboContextBlock, "_allowF16Attention"));
        Assert.False(GetPrivateField<bool>(baseMainBlock, "_useF16SandwichDamp"));
        Assert.Equal(DitDtype.Act == DType.F16,
            GetPrivateField<bool>(turboMainBlock, "_useF16SandwichDamp"));
    }

    [Fact]
    public void AlternatingPackedCaptions_RetainsBothRefinedStreams()
    {
        using CpuBackend cpu = new();
        ZImageConfig config = TinyConfig();
        Dictionary<string, Tensor> weights = TinyWeights(config);
        try
        {
            using ZImageTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor cond = RandomTensor(new TensorShape(1, 3, config.CapFeatDim), 1102);
            using Tensor uncond = RandomTensor(new TensorShape(1, 2, config.CapFeatDim), 1103);

            Tensor firstCond = EnsureRefinedCaption(transformer, cpu, cond);
            Tensor firstUncond = EnsureRefinedCaption(transformer, cpu, uncond);
            AssertFinite(firstCond);
            AssertFinite(firstUncond);

            List<(object Key, Tensor Value)> firstEntries = CachedCaptionEntries(transformer);
            Assert.Equal(2, firstEntries.Count);
            Tensor condRefined = Assert.Single(firstEntries, entry => ReferenceEquals(entry.Key, cond)).Value;
            Tensor uncondRefined = Assert.Single(firstEntries, entry => ReferenceEquals(entry.Key, uncond)).Value;

            Tensor secondCond = EnsureRefinedCaption(transformer, cpu, cond);
            Tensor secondUncond = EnsureRefinedCaption(transformer, cpu, uncond);
            Assert.Same(firstCond, secondCond);
            Assert.Same(firstUncond, secondUncond);

            List<(object Key, Tensor Value)> secondEntries = CachedCaptionEntries(transformer);
            Assert.Same(condRefined,
                Assert.Single(secondEntries, entry => ReferenceEquals(entry.Key, cond)).Value);
            Assert.Same(uncondRefined,
                Assert.Single(secondEntries, entry => ReferenceEquals(entry.Key, uncond)).Value);
        }
        finally
        {
            DisposeWeights(weights);
        }
    }

    [Fact]
    public void RefinedCaptionEviction_DisposeFailureCannotLeaveStaleHit()
    {
        using CpuBackend cpu = new();
        ZImageConfig config = TinyConfig();
        Dictionary<string, Tensor> weights = TinyWeights(config);
        try
        {
            using ZImageTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor first = RandomTensor(new TensorShape(1, 3, config.CapFeatDim), 1151);
            using Tensor second = RandomTensor(new TensorShape(1, 2, config.CapFeatDim), 1152);
            using Tensor replacement = RandomTensor(new TensorShape(1, 4, config.CapFeatDim), 1153);

            Tensor firstRefined = EnsureRefinedCaption(transformer, cpu, first);
            _ = EnsureRefinedCaption(transformer, cpu, second);
            InjectDisposeFailure(firstRefined);

            TargetInvocationException error = Assert.Throws<TargetInvocationException>(
                () => EnsureRefinedCaption(transformer, cpu, replacement));
            Assert.IsType<InvalidOperationException>(error.InnerException);

            List<(object Key, Tensor Value)> afterFailure = CachedCaptionEntries(transformer);
            Assert.Single(afterFailure);
            Assert.Same(second, afterFailure[0].Key);
            Assert.DoesNotContain(afterFailure, entry => ReferenceEquals(entry.Key, first));
            Assert.DoesNotContain(afterFailure, entry => ReferenceEquals(entry.Key, replacement));

            // The cleared slot remains usable after the failed disposal; the disposed old tensor is not returned.
            Tensor recomputed = EnsureRefinedCaption(transformer, cpu, first);
            Assert.NotSame(firstRefined, recomputed);
            Assert.Equal(2, CachedCaptionEntries(transformer).Count);
        }
        finally
        {
            DisposeWeights(weights);
        }
    }

    [Fact]
    public void TransformerDispose_AttemptsEveryOwnedTensorAfterFirstFailure()
    {
        using CpuBackend cpu = new();
        ZImageConfig config = TinyConfig();
        Dictionary<string, Tensor> weights = TinyWeights(config);
        ZImageTransformer transformer = new(config);
        Tensor? latentFixed = null;
        Tensor? timestepFixed = null;
        Tensor? graphVelocity = null;
        try
        {
            transformer.LoadWeights(weights);
            using Tensor first = RandomTensor(new TensorShape(1, 3, config.CapFeatDim), 1161);
            using Tensor second = RandomTensor(new TensorShape(1, 2, config.CapFeatDim), 1162);
            Tensor firstRefined = EnsureRefinedCaption(transformer, cpu, first);
            Tensor secondRefined = EnsureRefinedCaption(transformer, cpu, second);
            InjectDisposeFailure(firstRefined);

            latentFixed = RandomTensor(new TensorShape(1), 1163);
            timestepFixed = RandomTensor(new TensorShape(1), 1164);
            graphVelocity = RandomTensor(new TensorShape(1), 1165);
            SetPrivateField(transformer, "_latentFixed", latentFixed);
            SetPrivateField(transformer, "_tEmbFixed", timestepFixed);
            SetPrivateField(transformer, "_graphVelocity", graphVelocity);

            InvalidOperationException error = Assert.Throws<InvalidOperationException>(transformer.Dispose);
            Assert.IsType<InvalidOperationException>(error.InnerException);
            Assert.Empty(CachedCaptionEntries(transformer));
            AssertDisposed(firstRefined);
            AssertDisposed(secondRefined);
            AssertDisposed(latentFixed);
            AssertDisposed(timestepFixed);
            AssertDisposed(graphVelocity);
        }
        finally
        {
            // Dispose is idempotent after the fault and will not rethrow or skip weight-owner cleanup.
            transformer.Dispose();
            latentFixed?.Dispose();
            timestepFixed?.Dispose();
            graphVelocity?.Dispose();
            DisposeWeights(weights);
        }
    }

    [Fact]
    [Trait("Category", "Cuda")]
    public void WarmPackedCfgPairAndEuler_HasNoIntermediateD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        ZImageConfig config = TinyConfig();
        Dictionary<string, Tensor> weights = TinyWeights(config);
        using CudaBackend cuda = new(0, PtxDir());
        try
        {
            using ZImageTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            cuda.PreloadWeights(transformer.EnumerateWeights());

            using Tensor latentNchw = RandomTensor(new TensorShape(1, config.InChannels, 4, 4), 1201);
            using Tensor packed = transformer.PatchifyLatent(cuda, latentNchw);
            using Tensor cond = RandomTensor(new TensorShape(1, 3, config.CapFeatDim), 1202);
            using Tensor uncond = RandomTensor(new TensorShape(1, 2, config.CapFeatDim), 1203);

            // Populate both refined-caption entries and warm their device uploads before measuring the steady loop.
            using (Tensor warmCond = transformer.ForwardPacked(cuda, packed, cond, 0.2f, 2, 2)) { }
            using (Tensor warmUncond = transformer.ForwardPacked(cuda, packed, uncond, 0.2f, 2, 2)) { }
            cuda.Sync();
            cuda.ResetD2hSyncCount();

            using Tensor condVelocity = transformer.ForwardPacked(cuda, packed, cond, 0.6f, 2, 2);
            using Tensor uncondVelocity = transformer.ForwardPacked(cuda, packed, uncond, 0.6f, 2, 2);
            cuda.CfgEulerStep(packed, condVelocity, uncondVelocity, guidance: 5.0f, delta: 0.125f);
            cuda.Sync();
            Assert.Equal(0, cuda.GetD2hSyncCount());

            AssertFinite(packed); // sole intentional final readback
            Assert.Equal(1, cuda.GetD2hSyncCount());
            _output.WriteLine("Warm Z-Image packed cond + uncond + fused Euler: intermediate D2H=0.");

            transformer.ReleaseDeviceCache(cuda);
            cuda.FreeWeights(transformer.EnumerateWeights());
            cuda.FreeActivations();
        }
        finally
        {
            DisposeWeights(weights);
        }
    }

    private static ZImageConfig TinyConfig() => new()
    {
        HiddenSize = 8,
        NumHeads = 1,
        HeadDim = 8,
        NumLayers = 0,
        NumRefinerLayers = 0,
        InChannels = 2,
        PatchSize = 2,
        CapFeatDim = 4,
        AdaLNEmbedDim = 4,
        AxesDims = [2, 2, 4],
        AxesLens = [16, 16, 16],
        SeqMultiOf = 1,
        FfnDim = 12,
    };

    private static Dictionary<string, Tensor> TinyWeights(ZImageConfig config)
    {
        ZetaChromaConfig zeta = new()
        {
            Backbone = config,
            PatchSize = config.PatchSize,
            DecoderHidden = config.HiddenSize,
            DecoderResBlocks = 0,
            DecoderMaxFreqs = 1,
        };
        Dictionary<string, Tensor> weights = ZetaChromaSyntheticWeights.Build(zeta, tEmbMlpHidden: 8);
        int patchDim = config.InChannels * config.PatchSize * config.PatchSize;
        weights["final_layer.adaLN_modulation.1.weight"] = RandomTensor(
            new TensorShape(config.HiddenSize, config.AdaLNEmbedDim), 1301);
        weights["final_layer.adaLN_modulation.1.bias"] = RandomTensor(
            new TensorShape(config.HiddenSize), 1302);
        weights["final_layer.linear.weight"] = RandomTensor(
            new TensorShape(patchDim, config.HiddenSize), 1303);
        weights["final_layer.linear.bias"] = RandomTensor(new TensorShape(patchDim), 1304);
        return weights;
    }

    private static List<(object Key, Tensor Value)> CachedCaptionEntries(ZImageTransformer transformer)
    {
        FieldInfo cacheField = typeof(ZImageTransformer).GetField("_refinedCaptionCache",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Refined-caption cache field was not found.");
        Array entries = (Array)(cacheField.GetValue(transformer)
            ?? throw new InvalidOperationException("Refined-caption cache is null."));
        List<(object Key, Tensor Value)> result = [];
        foreach (object entry in entries)
        {
            Type type = entry.GetType();
            object? key = type.GetField("Key")!.GetValue(entry);
            Tensor? value = (Tensor?)type.GetField("Value")!.GetValue(entry);
            if (key is not null && value is not null)
                result.Add((key, value));
        }
        return result;
    }

    private static Tensor EnsureRefinedCaption(
        ZImageTransformer transformer, IBackend backend, Tensor caption)
    {
        MethodInfo method = typeof(ZImageTransformer).GetMethod("EnsureRefinedCaption",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("EnsureRefinedCaption method was not found.");
        return (Tensor)(method.Invoke(transformer,
            [backend, caption, 1, (int)caption.Shape[1], (int)caption.Shape[1], DType.F32])
            ?? throw new InvalidOperationException("EnsureRefinedCaption returned null."));
    }

    private static void InjectDisposeFailure(Tensor tensor)
    {
        FieldInfo field = typeof(Tensor).GetField("_gpuDisposeCallback",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Tensor GPU-dispose callback field was not found.");
        field.SetValue(tensor, (Action)(() => throw new InvalidOperationException("injected dispose failure")));
    }

    private static void SetPrivateField(ZImageTransformer transformer, string name, Tensor value)
    {
        FieldInfo field = typeof(ZImageTransformer).GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"ZImageTransformer field {name} was not found.");
        field.SetValue(transformer, value);
    }

    private static T GetPrivateField<T>(object instance, string name)
    {
        FieldInfo field = instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field {instance.GetType().Name}.{name} was not found.");
        return (T)(field.GetValue(instance)
            ?? throw new InvalidOperationException($"Field {instance.GetType().Name}.{name} is null."));
    }

    private static object[] GetPrivateArray(object instance, string name)
    {
        Array array = GetPrivateField<Array>(instance, name);
        return array.Cast<object>().ToArray();
    }

    private static void AssertDisposed(Tensor tensor)
    {
        Assert.Throws<ObjectDisposedException>(() => { _ = (nint)tensor.DataPointer; });
    }

    private static Tensor Values(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static Tensor RandomTensor(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        Random random = new(seed);
        for (long i = 0; i < tensor.ElementCount; i++)
            data[i] = (float)(random.NextDouble() * 0.1 - 0.05);
        return tensor;
    }

    private static void AssertClose(float[] expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.ElementCount);
        float* data = (float*)actual.DataPointer;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]), $"Expected value {i} is non-finite: {expected[i]}");
            Assert.True(float.IsFinite(data[i]), $"Actual value {i} is non-finite: {data[i]}");
            Assert.InRange(MathF.Abs(data[i] - expected[i]), 0f, tolerance);
        }
    }

    private static void AssertFinite(Tensor tensor)
    {
        float* data = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            Assert.True(float.IsFinite(data[i]), $"Non-finite value at element {i}: {data[i]}");
    }

    private static void DisposeWeights(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values.Distinct())
            tensor.Dispose();
    }

    private static string? PtxDir()
    {
        string path = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return Directory.Exists(path) ? path : null;
    }
}
