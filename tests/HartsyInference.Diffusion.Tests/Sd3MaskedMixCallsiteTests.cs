using System.Reflection;
using System.Runtime.ExceptionServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Locks SD3's three concrete mix callsites to their legacy scheduler/mask math and verifies that the CUDA path
/// preserves its pinned source/mask/latent across activation sweeps without an intermediate readback.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class Sd3MaskedMixCallsiteTests
{
    private readonly ITestOutputHelper _output;

    public Sd3MaskedMixCallsiteTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(3.0f, 28, 0)]
    [InlineData(3.0f, 28, 13)]
    [InlineData(5.0f, 11, 10)]
    public void InitialImg2ImgMix_UsesAffinePrimitive_AndExactlyMatchesLegacyAddNoise(
        float shift, int steps, int stepIndex)
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(shift);
        scheduler.SetTimesteps(steps);
        TensorShape shape = new(2, 3, 2, 5);
        float[] sourceValues = Values((int)shape.ElementCount, 101, 2f);
        float[] noiseValues = Values((int)shape.ElementCount, 103, 1f);

        using Tensor source = TensorFrom(sourceValues, shape);
        using Tensor noise = TensorFrom(noiseValues, shape);
        using Tensor expected = new(shape, DType.F32);
        scheduler.AddNoise(expected, source, noise, stepIndex);
        using Tensor actual = new(shape, DType.F32);
        using IBackend backend = RecordingBackendProxy.Create(out RecordingBackendProxy recording);

        Sd3Pipeline.AddFlowMatchNoise(backend, scheduler, actual, source, noise, stepIndex);

        Assert.Equal(1, recording.AffineMixCalls);
        Assert.Same(actual, recording.LastTarget);
        Assert.Same(source, recording.LastSource);
        Assert.Same(noise, recording.LastNoise);
        Assert.Equal(BitConverter.SingleToUInt32Bits(1f - scheduler.SigmaAt(stepIndex)),
            BitConverter.SingleToUInt32Bits(recording.LastSourceScale));
        Assert.Equal(BitConverter.SingleToUInt32Bits(scheduler.SigmaAt(stepIndex)),
            BitConverter.SingleToUInt32Bits(recording.LastNoiseScale));
        AssertExact(Snapshot(expected), Snapshot(actual), "initial AddNoise substitution");
        AssertExact(sourceValues, Snapshot(source), "initial source mutation");
        AssertExact(noiseValues, Snapshot(noise), "initial noise mutation");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(12)]
    public void PerStepMaskedMix_PreservesLegacySeedScheduleAndTerminalCleanSource(int nextStep)
    {
        const int steps = 12, seed = 424242;
        FlowMatchEulerDiscreteScheduler scheduler = new(3f);
        scheduler.SetTimesteps(steps);
        TensorShape shape = new(1, 4, 3, 5);
        TensorShape maskShape = new(1, 1, 3, 5);
        float[] initialValues = Values((int)shape.ElementCount, 211, 1.5f);
        float[] sourceValues = Values((int)shape.ElementCount, 223, 0.75f);
        float[] maskValues = MaskValues((int)maskShape.ElementCount);

        using Tensor source = TensorFrom(sourceValues, shape);
        using Tensor mask = TensorFrom(maskValues, maskShape);
        using Tensor expected = TensorFrom(initialValues, shape);
        ApplyLegacyMaskedStep(expected, source, mask, scheduler, seed, nextStep);
        using Tensor actual = TensorFrom(initialValues, shape);
        using IBackend backend = RecordingBackendProxy.Create(out RecordingBackendProxy recording);

        Sd3Pipeline.BlendMaskedSourceTrajectory(
            backend, scheduler, actual, source, mask, seed, nextStep);

        float sigma = scheduler.SigmaAt(nextStep);
        Assert.Equal(1, recording.MaskedMixCalls);
        Assert.Same(actual, recording.LastTarget);
        Assert.Same(source, recording.LastSource);
        Assert.Same(mask, recording.LastMask);
        Assert.Equal(MaskBroadcastLayout.DenseNchwBroadcast, recording.LastLayout);
        Assert.Equal(BitConverter.SingleToUInt32Bits(1f - sigma),
            BitConverter.SingleToUInt32Bits(recording.LastSourceScale));
        Assert.Equal(BitConverter.SingleToUInt32Bits(sigma),
            BitConverter.SingleToUInt32Bits(recording.LastNoiseScale));
        Assert.Equal(nextStep == steps, recording.LastNoiseWasNull);

        if (nextStep < steps)
        {
            using Tensor expectedFreshNoise = SeedGenerator.CreateNoise(shape, seed + nextStep);
            AssertExact(Snapshot(expectedFreshNoise), recording.LastNoiseSnapshot!, "seed + nextStep fresh noise");
        }
        else
        {
            Assert.Null(recording.LastNoiseSnapshot);
        }

        AssertExact(Snapshot(expected), Snapshot(actual), "per-step masked fusion");
        AssertExact(sourceValues, Snapshot(source), "per-step source mutation");
        AssertExact(maskValues, Snapshot(mask), "per-step mask mutation");
    }

    [Fact]
    public void PerStepMaskedMix_DisposesFreshNoise_WhenBackendThrows()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(3f);
        scheduler.SetTimesteps(4);
        TensorShape shape = new(1, 2, 2, 3);
        using Tensor target = TensorFrom(Values(12, 301, 1f), shape);
        using Tensor source = TensorFrom(Values(12, 307, 1f), shape);
        using Tensor mask = TensorFrom(MaskValues(6), new TensorShape(1, 1, 2, 3));
        using IBackend backend = RecordingBackendProxy.Create(out RecordingBackendProxy recording);
        recording.ThrowOnMaskedMix = true;

        Assert.Throws<InvalidOperationException>(() =>
            Sd3Pipeline.BlendMaskedSourceTrajectory(backend, scheduler, target, source, mask, seed: 17, nextStep: 2));
        Assert.NotNull(recording.LastNoise);
        Assert.Throws<ObjectDisposedException>(() => recording.LastNoise!.AsSpan<float>());
    }

    [Fact]
    public void FinalPixelRecompose_UsesDenseMaskedPrimitive_AndExactlyMatchesLegacyBlend()
    {
        TensorShape imageShape = new(1, 3, 4, 5);
        TensorShape maskShape = new(1, 1, 4, 5);
        float[] decodedValues = Values((int)imageShape.ElementCount, 401, 1f);
        float[] sourceValues = Values((int)imageShape.ElementCount, 409, 1f);
        float[] maskValues = MaskValues((int)maskShape.ElementCount);
        using Tensor source = TensorFrom(sourceValues, imageShape);
        using Tensor mask = TensorFrom(maskValues, maskShape);
        using Tensor expected = TensorFrom(decodedValues, imageShape);
        MaskBlendUtilities.BlendChannelsInPlace(expected, source, mask);
        using Tensor actual = TensorFrom(decodedValues, imageShape);
        using IBackend backend = RecordingBackendProxy.Create(out RecordingBackendProxy recording);

        Sd3Pipeline.RecomposeMaskedImage(backend, actual, source, mask);

        Assert.Equal(1, recording.MaskedMixCalls);
        Assert.True(recording.LastNoiseWasNull);
        Assert.Equal(1f, recording.LastSourceScale);
        Assert.Equal(0f, recording.LastNoiseScale);
        Assert.Equal(MaskBroadcastLayout.DenseNchwBroadcast, recording.LastLayout);
        AssertExact(Snapshot(expected), Snapshot(actual), "final pixel recomposite");
        AssertExact(sourceValues, Snapshot(source), "pixel source mutation");
        AssertExact(maskValues, Snapshot(mask), "pixel mask mutation");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_MaskedCallsites_SurviveActivationSweeps_WithoutIntermediateD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int steps = 8, initialStep = 3, maskedNextStep = 4, seed = 5150;
        FlowMatchEulerDiscreteScheduler scheduler = new(3f);
        scheduler.SetTimesteps(steps);
        TensorShape shape = new(1, 4, 5, 7);
        TensorShape maskShape = new(1, 1, 5, 7);
        float[] sourceValues = Values((int)shape.ElementCount, 503, 1.25f);
        float[] initialNoiseValues = Values((int)shape.ElementCount, 509, 0.8f);
        float[] maskValues = MaskValues((int)maskShape.ElementCount);

        using Tensor sourceHost = TensorFrom(sourceValues, shape);
        using Tensor noiseHost = TensorFrom(initialNoiseValues, shape);
        using Tensor maskHost = TensorFrom(maskValues, maskShape);
        using Tensor expected = new(shape, DType.F32);
        scheduler.AddNoise(expected, sourceHost, noiseHost, initialStep);
        ApplyLegacyMaskedStep(expected, sourceHost, maskHost, scheduler, seed, maskedNextStep);
        ApplyLegacyMaskedStep(expected, sourceHost, maskHost, scheduler, seed, steps);

        using Tensor source = new(shape, DType.F32);
        using Tensor mask = new(maskShape, DType.F32);
        using Tensor latent = new(shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(source, sourceHost, 1f);
        cuda.Scale(mask, maskHost, 1f);

        bool sourcePinned = false, maskPinned = false, latentPinned = false;
        try
        {
            cuda.PinActivation(source);
            sourcePinned = true;
            cuda.PinActivation(mask);
            maskPinned = true;
            cuda.FreeActivations(trimPool: false);
            cuda.ResetD2hSyncCount();

            Sd3Pipeline.AddFlowMatchNoise(cuda, scheduler, latent, source, noiseHost, initialStep);
            cuda.PinActivation(latent);
            latentPinned = true;
            cuda.FreeActivations(trimPool: false);
            Sd3Pipeline.BlendMaskedSourceTrajectory(
                cuda, scheduler, latent, source, mask, seed, maskedNextStep);
            cuda.FreeActivations(trimPool: false);
            Sd3Pipeline.BlendMaskedSourceTrajectory(cuda, scheduler, latent, source, mask, seed, steps);
            cuda.Sync();
            Assert.Equal(0, cuda.GetD2hSyncCount());
        }
        finally
        {
            if (latentPinned) cuda.UnpinActivation(latent);
            if (maskPinned) cuda.UnpinActivation(mask);
            if (sourcePinned) cuda.UnpinActivation(source);
        }

        AssertClose(Snapshot(expected), Snapshot(latent), 4e-6f, "resident SD3 masked chain");
        Assert.Equal(1, cuda.GetD2hSyncCount());
        AssertExact(sourceValues, Snapshot(source), "resident source mutation");
        AssertExact(maskValues, Snapshot(mask), "resident mask mutation");
    }

    private static void ApplyLegacyMaskedStep(
        Tensor target,
        Tensor source,
        Tensor mask,
        FlowMatchEulerDiscreteScheduler scheduler,
        int seed,
        int nextStep)
    {
        if (nextStep < scheduler.NumInferenceSteps)
        {
            using Tensor freshNoise = SeedGenerator.CreateNoise(target.Shape, seed + nextStep);
            using Tensor noisedSource = new(target.Shape, DType.F32);
            scheduler.AddNoise(noisedSource, source, freshNoise, nextStep);
            MaskBlendUtilities.BlendChannelsInPlace(target, noisedSource, mask);
        }
        else
        {
            MaskBlendUtilities.BlendChannelsInPlace(target, source, mask);
        }
    }

    private static float[] Values(int count, int seed, float scale)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = ((float)random.NextDouble() * 2f - 1f) * scale;
        return values;
    }

    private static float[] MaskValues(int count) =>
        Enumerable.Range(0, count).Select(i => (i % 7) / 6f).ToArray();

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor) => tensor.AsReadOnlySpan<float>().ToArray();

    private static void AssertExact(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(actual[i]),
                $"{label}: index {i}, expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(error <= tolerance || error <= tolerance * MathF.Max(1f, MathF.Abs(expected[i])),
                $"{label}: index {i}, expected {expected[i]:R}, got {actual[i]:R}, error {error:R}");
        }
    }

    /// <summary>Call-recording interface proxy that delegates all math to the real CPU backend.</summary>
    public class RecordingBackendProxy : DispatchProxy
    {
        private IBackend _inner = null!;

        public int AffineMixCalls { get; private set; }
        public int MaskedMixCalls { get; private set; }
        public Tensor? LastTarget { get; private set; }
        public Tensor? LastSource { get; private set; }
        public Tensor? LastNoise { get; private set; }
        public Tensor? LastMask { get; private set; }
        public float LastSourceScale { get; private set; }
        public float LastNoiseScale { get; private set; }
        public MaskBroadcastLayout LastLayout { get; private set; }
        public bool LastNoiseWasNull { get; private set; }
        public float[]? LastNoiseSnapshot { get; private set; }
        public bool ThrowOnMaskedMix { get; set; }

        public static IBackend Create(out RecordingBackendProxy recording)
        {
            IBackend backend = DispatchProxy.Create<IBackend, RecordingBackendProxy>();
            recording = (RecordingBackendProxy)backend;
            recording._inner = new CpuBackend();
            return backend;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            args ??= [];
            if (targetMethod.Name == nameof(IBackend.AffineMix))
            {
                AffineMixCalls++;
                LastTarget = (Tensor)args[0]!;
                LastSource = (Tensor)args[1]!;
                LastNoise = (Tensor)args[2]!;
                LastSourceScale = (float)args[3]!;
                LastNoiseScale = (float)args[4]!;
            }
            else if (targetMethod.Name == nameof(IBackend.MaskedAffineMixInPlace))
            {
                MaskedMixCalls++;
                LastTarget = (Tensor)args[0]!;
                LastSource = (Tensor)args[1]!;
                LastNoise = (Tensor?)args[2];
                LastMask = (Tensor)args[3]!;
                LastSourceScale = (float)args[4]!;
                LastNoiseScale = (float)args[5]!;
                LastLayout = (MaskBroadcastLayout)args[6]!;
                LastNoiseWasNull = LastNoise is null;
                LastNoiseSnapshot = LastNoise is null ? null : Snapshot(LastNoise);
                if (ThrowOnMaskedMix)
                    throw new InvalidOperationException("Injected masked-mix failure.");
            }

            try
            {
                return targetMethod.Invoke(_inner, args);
            }
            catch (TargetInvocationException error) when (error.InnerException is not null)
            {
                ExceptionDispatchInfo.Capture(error.InnerException).Throw();
                throw;
            }
        }
    }
}
