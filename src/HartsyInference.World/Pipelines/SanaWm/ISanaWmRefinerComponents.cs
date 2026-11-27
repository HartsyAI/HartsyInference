using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The text path of SANA-WM's refiner: the prompt through Gemma-3-12B and the LTX-2.0 text connector.</summary>
public interface ISanaWmRefinerTextEncoder
{
    /// <summary>The refiner's prompt embedding for <paramref name="prompt"/>, taken as written (the refiner has no prompt
    /// prefix and no negative prompt): host F32 <c>[L, 3840]</c>, which the caller owns.</summary>
    Tensor Encode(string prompt);
}

/// <summary>How a stream's refiner stage runs. The defaults are upstream's (<c>--refiner_seed 42</c>,
/// <c>--refiner_kv_max_frames 11</c>).</summary>
public sealed record SanaWmRefinerSettings
{
    /// <summary>Seed of the refiner's per-block noise.</summary>
    public long Seed { get; init; } = 42;

    /// <summary>Most latent frames the refiner's attention window holds, the seed frame included. A smaller window needs less
    /// memory; below the seed frame plus one chunk the decoded video flickers at chunk boundaries.</summary>
    public int WindowFrames { get; init; } = 11;

    /// <summary>Keeps the sliding history on every N-th transformer block only (and the last). 1 is the trained setup; a
    /// larger stride divides the window's memory and is an approximation.</summary>
    public int HistoryLayerStride { get; init; } = 1;

    /// <summary>Replaces the seeded noise of a refiner block, by block index, with a host F32 <c>[1, 128, T, 22, 40]</c>
    /// tensor borrowed for the call (parity against a recorded upstream run); a null result draws the seeded noise.</summary>
    public Func<int, Tensor?>? Noise { get; init; }

    /// <summary>Observes the refiner's intermediates by name (see <see cref="LtxVideo2ChunkRefinerOptions.Tap"/>).</summary>
    public Action<string, Tensor>? Tap { get; init; }

    /// <summary>The refiner options of a SANA-WM stream: the model's latent grid at its frame rate, one sink frame (the seed
    /// frame), upstream's three-step schedule, and these settings.</summary>
    /// <param name="headroomBytes">Device bytes the refiner's block weights leave free beyond its attention window.</param>
    public LtxVideo2ChunkRefinerOptions ToOptions(long headroomBytes) => new()
    {
        LatentHeight = SanaWmPipeline.LatentHeight,
        LatentWidth = SanaWmPipeline.LatentWidth,
        Fps = SanaWmAction.Fps,
        Window = new LtxVideo2ChunkWindow { MaxFrames = WindowFrames, HistoryLayerStride = HistoryLayerStride },
        Seed = Seed,
        Noise = Noise,
        Tap = Tap,
        HeadroomBytes = headroomBytes,
    };
}

/// <summary>Loads the components of a SANA-WM stream's refiner stage. A component set that also implements this can run
/// streams with <see cref="SanaWmStreamRequest.Refine"/>. The refiner's text encoder is opened and released before the DiT
/// loads; the refiner itself is opened last and stays resident beside the DiT and the decoder for the whole stream.</summary>
public interface ISanaWmRefinerComponents
{
    /// <summary>Loads Gemma-3-12B, its tokenizer and the LTX-2.0 text connector.</summary>
    SanaWmStage<ISanaWmRefinerTextEncoder> OpenRefinerTextEncoder();

    /// <summary>Loads the chunk-causal LTX-2 refiner, bound to the stream's prompt embedding
    /// (<paramref name="promptEmbeds"/>, borrowed for the call).</summary>
    SanaWmStage<IStreamingLatentRefiner> OpenRefiner(Tensor promptEmbeds, SanaWmRefinerSettings settings);
}
