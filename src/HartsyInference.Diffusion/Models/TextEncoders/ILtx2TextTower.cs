using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>The conditioning text tower an LTX-2 pipeline drives. LTX-2.3 uses Gemma 3 12B
/// (<see cref="LlamaStyleEncoder"/>); LTX-2.5 changed family to a Gemma 4 12B fine-tune
/// (<see cref="Gemma4TextEncoder"/>) whose per-layer geometry cannot be expressed as a
/// <see cref="LlamaStyleEncoderConfig"/> preset.</summary>
/// <remarks>The pipeline's <c>3840</c> caption channels and <c>49</c> harvested states hold for both families.</remarks>
public interface ILtx2TextTower : IDisposable
{
    /// <summary>Transformer block count; <see cref="EncodeMultiLayer"/> accepts state indices <c>0 … NumLayers</c>.</summary>
    int NumLayers { get; }

    /// <summary>Harvests the requested hidden states for each prompt, concatenated on the last dim.</summary>
    /// <param name="layerIndices">State indices to keep, where 0 is the embedding output and <see cref="NumLayers"/> the last block.</param>
    /// <param name="interleavedLayout">Interleave per position instead of the default layer-outer <c>[B, seq, K·H]</c>.</param>
    Tensor EncodeMultiLayer(IBackend backend, int[][] tokenIds, int[] layerIndices, bool interleavedLayout = false);

    /// <summary>Resident weight tensors, for GPU preload and for the pipeline's own VRAM accounting.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
