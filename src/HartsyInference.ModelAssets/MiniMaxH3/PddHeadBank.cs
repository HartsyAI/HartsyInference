using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Validated video/audio parallel-decoder heads, normalized to one row view per fine interval.</summary>
public sealed unsafe class PddHeadBank : IDisposable
{
    /// <summary>Published H3 video projection width after 1x2x2 patchification.</summary>
    public const int PublishedVideoChannels = 96;

    /// <summary>Published H3 audio projection width.</summary>
    public const int PublishedAudioChannels = 32;

    /// <summary>Published H3 transformer width.</summary>
    public const int PublishedHiddenSize = 5376;

    private readonly Tensor[] _videoWeights;
    private readonly Tensor[] _videoBiases;
    private readonly Tensor[] _audioWeights;
    private readonly Tensor[] _audioBiases;
    private int _disposed;

    /// <summary>Validates either the official rank-three bank or a hash/metadata-confirmed flattened bank.</summary>
    public PddHeadBank(Tensor videoWeight, Tensor videoBias, Tensor audioWeight, Tensor audioBias,
        MiniMaxH3PddHeadLayout layout, bool requirePublishedShape = true)
    {
        ArgumentNullException.ThrowIfNull(videoWeight);
        ArgumentNullException.ThrowIfNull(videoBias);
        ArgumentNullException.ThrowIfNull(audioWeight);
        ArgumentNullException.ThrowIfNull(audioBias);
        if (videoWeight.DType != DType.F32 || videoBias.DType != DType.F32
            || audioWeight.DType != DType.F32 || audioBias.DType != DType.F32)
        {
            throw new HartsyInferenceException(
                "MiniMax-H3 PDD final-head banks must be F32; converting a multi-gigabyte bank implicitly is unsafe.");
        }

        (int videoSteps, int videoOut, int hidden) = WeightGeometry(videoWeight, "video");
        (int audioSteps, int audioOut, int audioHidden) = WeightGeometry(audioWeight, "audio");
        (int videoBiasSteps, int videoBiasOut) = BiasGeometry(videoBias, "video");
        (int audioBiasSteps, int audioBiasOut) = BiasGeometry(audioBias, "audio");
        if (videoSteps != audioSteps || videoSteps != videoBiasSteps || videoSteps != audioBiasSteps)
        {
            throw new HartsyInferenceException(
                $"PDD head step counts disagree: video={videoSteps}, audio={audioSteps}, "
                + $"videoBias={videoBiasSteps}, audioBias={audioBiasSteps}.");
        }
        if (videoSteps != MiniMaxH3PddSchedule.PublishedFineSteps)
        {
            throw new HartsyInferenceException(
                $"Published MiniMax-H3 PDD banks require 32 heads; got {videoSteps}.");
        }
        if (videoOut != videoBiasOut || audioOut != audioBiasOut || hidden != audioHidden)
            throw new HartsyInferenceException("PDD weight/bias projection dimensions do not agree.");
        if (requirePublishedShape && (videoOut != PublishedVideoChannels || audioOut != PublishedAudioChannels
            || hidden != PublishedHiddenSize))
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 PDD expects video/audio/hidden dimensions "
                + $"{PublishedVideoChannels}/{PublishedAudioChannels}/{PublishedHiddenSize}; "
                + $"got {videoOut}/{audioOut}/{hidden}.");
        }

        Layout = layout;
        StepCount = videoSteps;
        VideoChannels = videoOut;
        AudioChannels = audioOut;
        HiddenSize = hidden;
        _videoWeights = CreateWeightViews(videoWeight, videoSteps, videoOut, hidden);
        _videoBiases = CreateBiasViews(videoBias, videoSteps, videoOut);
        _audioWeights = CreateWeightViews(audioWeight, audioSteps, audioOut, hidden);
        _audioBiases = CreateBiasViews(audioBias, audioSteps, audioOut);
    }

    /// <summary>Whether the source rows are complete heads or base-plus-offset rows.</summary>
    public MiniMaxH3PddHeadLayout Layout { get; }

    /// <summary>Number of fine intervals represented by the bank.</summary>
    public int StepCount { get; }

    /// <summary>Video head output width.</summary>
    public int VideoChannels { get; }

    /// <summary>Audio head output width.</summary>
    public int AudioChannels { get; }

    /// <summary>Shared input width of both heads.</summary>
    public int HiddenSize { get; }

    /// <summary>Every immutable tensor view that should be bulk-preloaded on the execution backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        ThrowIfDisposed();
        for (int i = 0; i < StepCount; i++)
        {
            yield return _videoWeights[i];
            yield return _videoBiases[i];
            yield return _audioWeights[i];
            yield return _audioBiases[i];
        }
    }

    /// <summary>Returns one video weight row without copying its mmap-backed bytes.</summary>
    public Tensor GetVideoWeight(int index) => Get(_videoWeights, index);

    /// <summary>Returns one video bias row without copying its mmap-backed bytes.</summary>
    public Tensor GetVideoBias(int index) => Get(_videoBiases, index);

    /// <summary>Returns one audio weight row without copying its mmap-backed bytes.</summary>
    public Tensor GetAudioWeight(int index) => Get(_audioWeights, index);

    /// <summary>Returns one audio bias row without copying its mmap-backed bytes.</summary>
    public Tensor GetAudioBias(int index) => Get(_audioBiases, index);

    private Tensor Get(Tensor[] rows, int index)
    {
        ThrowIfDisposed();
        if ((uint)index >= (uint)rows.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        return rows[index];
    }

    private static (int Steps, int Out, int Hidden) WeightGeometry(Tensor tensor, string name)
    {
        if (tensor.Shape.Rank == 3)
            return ((int)tensor.Shape[0], (int)tensor.Shape[1], (int)tensor.Shape[2]);
        if (tensor.Shape.Rank == 2)
        {
            int output = name == "video" ? PublishedVideoChannels : PublishedAudioChannels;
            if (tensor.Shape[0] % output != 0)
            {
                throw new HartsyInferenceException(
                    $"Flattened PDD {name} bank first dimension {tensor.Shape[0]} is not divisible by {output}.");
            }
            return ((int)(tensor.Shape[0] / output), output, (int)tensor.Shape[1]);
        }
        throw new HartsyInferenceException(
            $"PDD {name} head weight must be rank 3 or flattened rank 2; got {tensor.Shape}.");
    }

    private static (int Steps, int Out) BiasGeometry(Tensor tensor, string name)
    {
        if (tensor.Shape.Rank == 2)
            return ((int)tensor.Shape[0], (int)tensor.Shape[1]);
        if (tensor.Shape.Rank == 1)
        {
            int output = name == "video" ? PublishedVideoChannels : PublishedAudioChannels;
            if (tensor.Shape[0] % output != 0)
            {
                throw new HartsyInferenceException(
                    $"Flattened PDD {name} bias length {tensor.Shape[0]} is not divisible by {output}.");
            }
            return ((int)(tensor.Shape[0] / output), output);
        }
        throw new HartsyInferenceException(
            $"PDD {name} head bias must be rank 2 or flattened rank 1; got {tensor.Shape}.");
    }

    private static Tensor[] CreateWeightViews(Tensor source, int steps, int output, int hidden)
    {
        Tensor[] rows = new Tensor[steps];
        long rowBytes = DType.F32.ComputeByteCount((long)output * hidden);
        byte* sourcePointer = (byte*)source.DataPointer;
        for (int i = 0; i < steps; i++)
        {
            rows[i] = new Tensor(sourcePointer + i * rowBytes, new TensorShape(output, hidden), DType.F32);
        }
        return rows;
    }

    private static Tensor[] CreateBiasViews(Tensor source, int steps, int output)
    {
        Tensor[] rows = new Tensor[steps];
        long rowBytes = DType.F32.ComputeByteCount(output);
        byte* sourcePointer = (byte*)source.DataPointer;
        for (int i = 0; i < steps; i++)
        {
            rows[i] = new Tensor(sourcePointer + i * rowBytes, new TensorShape(output), DType.F32);
        }
        return rows;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases the borrowed row views; the source safetensors owner remains caller-owned.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in _videoWeights) tensor.Dispose();
        foreach (Tensor tensor in _videoBiases) tensor.Dispose();
        foreach (Tensor tensor in _audioWeights) tensor.Dispose();
        foreach (Tensor tensor in _audioBiases) tensor.Dispose();
    }
}
