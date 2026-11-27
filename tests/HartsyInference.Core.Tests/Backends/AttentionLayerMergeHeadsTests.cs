using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary><see cref="AttentionLayer.MergeHeads"/> hands back a new token-major tensor and takes over its input. When the backend's
/// permute throws, both the input and the new tensor must be released and the exception must reach the caller.
/// <para>The throw comes from a real backend that has been disposed (every operation of the CPU and CUDA backends refuses to run
/// then), so no fault-injection hook is involved. Only the input can be seen disposed from outside; the new tensor is internal to
/// the call, so the third test observes it through a thin forwarding proxy that records the output tensor the permute was handed.
/// A tensor is disposed when reading its data throws <see cref="ObjectDisposedException"/>.</para></summary>
public sealed class AttentionLayerMergeHeadsTests(ITestOutputHelper output)
{
    private const int Heads = 3, HeadDim = 5, Tokens = 7;

    private static Tensor HeadMajor(int batch)
    {
        Tensor tensor = new(new TensorShape(batch, Heads, Tokens, HeadDim), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = i + 1;
        }

        return tensor;
    }

    private static bool IsDisposed(Tensor tensor)
    {
        try
        {
            _ = tensor.AsReadOnlySpan<float>();
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    [Fact]
    public void MergeHeadsPermutesAndDisposesItsInput()
    {
        using CpuBackend backend = new();
        AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);
        Tensor input = HeadMajor(2);
        using Tensor merged = layer.MergeHeads(backend, input, new TensorShape(2, Tokens, Heads * HeadDim));

        Assert.True(IsDisposed(input));
        ReadOnlySpan<float> actual = merged.AsReadOnlySpan<float>();
        for (int b = 0; b < 2; b++)
        {
            for (int h = 0; h < Heads; h++)
            {
                for (int t = 0; t < Tokens; t++)
                {
                    for (int d = 0; d < HeadDim; d++)
                    {
                        float expected = (((b * Heads + h) * Tokens + t) * HeadDim + d) + 1;
                        Assert.Equal(expected, actual[((b * Tokens + t) * Heads + h) * HeadDim + d]);
                    }
                }
            }
        }
    }

    [Fact]
    public void WhenTheCpuBackendThrowsTheInputIsDisposedAndTheExceptionPropagates()
    {
        CpuBackend backend = new();
        backend.Dispose();
        AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);
        Tensor input = HeadMajor(1);

        Assert.Throws<ObjectDisposedException>(() => layer.MergeHeads(backend, input, new TensorShape(1, Tokens, Heads * HeadDim)));
        Assert.True(IsDisposed(input), "the input was not disposed after the permute threw");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WhenTheCudaBackendThrowsTheInputIsDisposedAndTheExceptionPropagates()
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        CudaBackend backend = new(0, PtxDirectory());
        backend.Dispose();
        AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);
        Tensor input = HeadMajor(1);

        Exception thrown = Assert.ThrowsAny<Exception>(() => layer.MergeHeads(backend, input, new TensorShape(1, Tokens, Heads * HeadDim)));
        output.WriteLine($"the disposed CUDA backend's permute threw {thrown.GetType().Name}");
        Assert.True(IsDisposed(input), "the input was not disposed after the permute threw");
    }

    /// <summary>Forwards every call to a real CPU backend except the permute, which records its output tensor and throws.</summary>
    public class RecordingBackend : DispatchProxy
    {
        public IBackend Target { get; set; } = null!;

        public Tensor? PermuteOutput { get; private set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod!.Name == nameof(IBackend.Permute0213))
            {
                PermuteOutput = (Tensor)args![0]!;
                throw new InvalidOperationException("the permute was refused");
            }

            try
            {
                return targetMethod.Invoke(Target, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }
    }

    [Fact]
    public void WhenThePermuteThrowsTheNewTensorIsDisposedToo()
    {
        using CpuBackend cpu = new();
        IBackend proxy = DispatchProxy.Create<IBackend, RecordingBackend>();
        ((RecordingBackend)proxy).Target = cpu;
        AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);
        Tensor input = HeadMajor(1);

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => layer.MergeHeads(proxy, input, new TensorShape(1, Tokens, Heads * HeadDim)));
        Assert.Equal("the permute was refused", ex.Message);
        Tensor created = ((RecordingBackend)proxy).PermuteOutput ?? throw new Xunit.Sdk.XunitException("the permute was never reached");
        Assert.True(IsDisposed(created), "the new tensor was not disposed after the permute threw");
        Assert.True(IsDisposed(input), "the input was not disposed after the permute threw");
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx was not found above the test directory.");
    }
}
