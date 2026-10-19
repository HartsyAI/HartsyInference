using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.Rope;

/// <summary>Backend stub for RoPE cache tests. Records preload, free and step-graph calls in order, and can fail the Nth
/// preload or every free on demand. Compute methods are unused by the cache.</summary>
internal sealed class RopeRecordingBackend : IBackend
{
    /// <summary>Every recorded call in order: <c>preload:N</c>, <c>free:N</c>, <c>free-failed:N</c>, <c>graph-reset</c>.</summary>
    public List<string> Calls { get; } = new();

    /// <summary>1-based index of the preload call that throws, or 0 for none.</summary>
    public int FailPreloadCall { get; set; }

    /// <summary>When set, <c>FreeWeights</c> throws after recording the attempt.</summary>
    public bool FailFree { get; set; }

    private int _preloadCalls;

    public IStreamingWeightCache? StreamingCache => null;

    public void TrimMemoryPool() { }

    public long FreeMemoryBytes() => long.MaxValue;

    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        _preloadCalls++;
        Calls.Add($"preload:{weights.Count()}");
        if (_preloadCalls == FailPreloadCall)
            throw new InvalidOperationException("Injected preload failure.");
    }

    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        int count = weights.Count();
        if (FailFree)
        {
            Calls.Add($"free-failed:{count}");
            throw new InvalidOperationException("Injected free failure.");
        }
        Calls.Add($"free:{count}");
    }

    public bool StepGraphSupported => true;

    public bool StepGraphReady { get; set; }

    public object? StepGraphOwner { get; set; }

    public void StepGraphReset()
    {
        StepGraphReady = false;
        Calls.Add("graph-reset");
    }

    public DeviceKind Device => DeviceKind.Cpu;

    public BackendCapabilities Capabilities { get; } = new BackendCapabilities { Name = "rope-recording-stub" };

    public void Dispose() { }

    public void MatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias) => throw Unused();
    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW) => throw Unused();
    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) => throw Unused();
    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps) => throw Unused();
    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps) => throw Unused();
    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) => throw Unused();
    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false) => throw Unused();
    public void Gelu(Tensor output, Tensor input) => throw Unused();
    public void Silu(Tensor output, Tensor input) => throw Unused();
    public void Sigmoid(Tensor output, Tensor input) => throw Unused();
    public void Tanh(Tensor output, Tensor input) => throw Unused();
    public void Elu(Tensor output, Tensor input, float alpha) => throw Unused();
    public void LeakyRelu(Tensor output, Tensor input, float slope) => throw Unused();
    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta) => throw Unused();
    public void Add(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Mul(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Scale(Tensor output, Tensor input, float scalar) => throw Unused();
    public void Clamp(Tensor output, Tensor input, float min, float max) => throw Unused();
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2) => throw Unused();
    public void Permute0213(Tensor output, Tensor input, int s, int h, int d) => throw Unused();
    public void GeGlu(Tensor output, Tensor input) => throw Unused();
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => throw Unused();
    public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim) => throw Unused();
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim) => throw Unused();
    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void CopyTo(Tensor destination, Tensor source) => throw Unused();
    public void Fill(Tensor tensor, float value) => throw Unused();
    public void Fft(Tensor output, Tensor input) => throw Unused();
    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => throw Unused();
    public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => throw Unused();

    private static Exception Unused([CallerMemberName] string member = "") =>
        new NotSupportedException($"{member} is not reachable from the RoPE table cache.");
}
