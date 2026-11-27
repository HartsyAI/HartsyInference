using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The multi-speaker part of a LongCat block: the audio step with the L-RoPE tables its own reference map gives.</summary>
public sealed partial class LongCatBlock
{
    /// <summary>The audio step of a multi-speaker forward. <paramref name="map"/> is the block's reference map <c>[regions, noisy tokens]</c>; the
    /// noisy tokens take their 1D positions from it. The probe receives the map, the positions and the cross-attention output.</summary>
    private Tensor SpeakerAudio(IBackend backend, Tensor stream, Tensor audio, Tensor timeEmbed, int condFrames, int tokensPerFrame,
        LongCatSpeakerPass speakers, float[] map, Action<string, Tensor>? probe, string stage)
    {
        int regions = speakers.Speakers.Regions, tokens = map.Length / regions;
        float[] positions = LongCatSpeakerRope.QueryPositions(map, regions, tokens);
        if (probe is not null)
        {
            ProbeHost(probe, $"{stage}.ref_attn_map", map, regions, tokens);
            ProbeHost(probe, $"{stage}.query_positions", positions, positions.Length);
            ProbeHost(probe, $"{stage}.key_positions", speakers.KeyPositions, speakers.KeyPositions.Length);
        }
        LongCatSpeakerRotation rotation = speakers.Rotation(positions);
        using (rotation.QueryCos)
        using (rotation.QuerySin)
            return _audio!.Apply(backend, stream, audio, timeEmbed, condFrames, tokensPerFrame, rotation, probe, $"{stage}.audio_cross_attn",
                roundSubLayer: _bf16Residual);
    }

    private static void ProbeHost(Action<string, Tensor> probe, string name, float[] values, params long[] shape)
    {
        using Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        probe(name, tensor);
    }

    /// <summary>The first <paramref name="rows"/> token-major rows of <paramref name="source"/> as head-major <c>[1, heads, rows,
    /// headDim]</c>.</summary>
    private Tensor HeadMajorFirstRows(IBackend backend, Tensor source, int rows)
    {
        using Tensor first = new(new TensorShape(rows, _config.HiddenSize), DType.F32);
        backend.SliceRows(first, source, 0);
        return HeadMajor(backend, first);
    }
}
