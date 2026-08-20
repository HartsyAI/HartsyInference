using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Backend stub that records the weight/pool calls <see cref="BlockStreamingScope"/> issues and serves a
/// scripted <see cref="IBackend.FreeMemoryBytes"/> sequence, so a placement decision is assertable without a GPU.</summary>
internal sealed class RecordingStreamingBackend : IBackend
{
    private readonly Queue<long> _freeReadings = new();
    private long _lastFreeReading;

    public RecordingStreamingBackend(IStreamingWeightCache? cache, params long[] freeReadings)
    {
        StreamingCache = cache;
        foreach (long reading in freeReadings) _freeReadings.Enqueue(reading);
        _lastFreeReading = freeReadings.Length > 0 ? freeReadings[^1] : 0;
    }

    /// <summary>Every recorded call in order, e.g. <c>trim</c>, <c>preload:a,b</c>, <c>free:c</c>, <c>freeBytes:1024</c>.</summary>
    public List<string> Calls { get; } = new();

    public IStreamingWeightCache? StreamingCache { get; }

    public void TrimMemoryPool() => Calls.Add("trim");

    /// <summary>Successive calls drain the scripted readings, then repeat the last one — a sizing that reads more
    /// often than the script expects is visible as a repeated value rather than an exception.</summary>
    public long FreeMemoryBytes()
    {
        if (_freeReadings.Count > 0) _lastFreeReading = _freeReadings.Dequeue();
        Calls.Add($"freeBytes:{_lastFreeReading}");
        return _lastFreeReading;
    }

    public void PreloadWeights(IEnumerable<Tensor> weights) => Calls.Add($"preload:{Describe(weights)}");

    public void FreeWeights(IEnumerable<Tensor> weights) => Calls.Add($"free:{Describe(weights)}");

    public DeviceKind Device => DeviceKind.Cpu;

    public BackendCapabilities Capabilities { get; } = new BackendCapabilities { Name = "recording-streaming-stub" };

    public void Dispose() { }

    private static string Describe(IEnumerable<Tensor> weights) => string.Join(",", weights.Select(TaggedStreamingBlock.IdFor));

    #region Unused compute surface

    private static Exception Unused([CallerMemberName] string member = "") =>
        new NotSupportedException($"{member} is not reachable from BlockStreamingScope.");

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

    #endregion
}
