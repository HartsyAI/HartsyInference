using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>A second-stage model that refines a stream of video latents a block at a time, as a first stage produces them:
/// each block is refined against what the refiner keeps of the blocks before it, so latents can be decoded while later
/// ones are still to come. The stream's first frames are the refiner's sink: they anchor every block and pass through
/// unrefined.</summary>
/// <remarks>A refiner follows the streaming state discipline (<see cref="IStreamingState"/>): <see cref="Refine"/> reads the
/// committed state and stages what the block adds, and <see cref="Commit"/> advances the refiner. A block that fails or is
/// cancelled is dropped with <see cref="Discard"/> and can be refined again. Called from one thread.</remarks>
public interface IStreamingLatentRefiner : IDisposable
{
    /// <summary>Leading frames of the stream that anchor the refiner and are not refined.</summary>
    int SinkFrames { get; }

    /// <summary>Whether <see cref="SetSink"/> has run.</summary>
    bool HasSink { get; }

    /// <summary>Frames refined and committed so far, the sink not counted.</summary>
    int RefinedFrames { get; }

    /// <summary>Model forwards one <see cref="Refine"/> call runs, for progress reporting.</summary>
    int PassesPerBlock { get; }

    /// <summary>Gives the refiner the stream's sink: host F32 <c>[1, C, SinkFrames, H, W]</c>, borrowed. Called once, before the
    /// first block. Takes effect as a whole or not at all.</summary>
    void SetSink(Tensor sinkLatents, CancellationToken cancel);

    /// <summary>Refines the next block: <paramref name="latents"/> is the first stage's clean latents for the frames that
    /// follow the refined ones, host F32 <c>[1, C, T, H, W]</c>, borrowed. Returns the refined latents in the same shape, which
    /// the caller owns. The refiner does not advance until <see cref="Commit"/>.</summary>
    /// <param name="onPass">Called after each model forward with its index within the block.</param>
    Tensor Refine(Tensor latents, CancellationToken cancel, Action<int>? onPass = null);

    /// <summary>Advances the refiner past the block the last <see cref="Refine"/> call produced.</summary>
    void Commit();

    /// <summary>Drops what the last <see cref="Refine"/> call staged; the refiner stays where it was.</summary>
    void Discard();
}
