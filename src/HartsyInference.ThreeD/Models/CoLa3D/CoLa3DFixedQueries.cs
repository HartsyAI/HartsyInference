using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Initial query tokens for <see cref="CoLa3DConfig.QuerySourceKind.Fixed"/>: a positional slot embedding through a
/// two-layer MLP, or a learned table. Borrows its weights.</summary>
internal sealed class CoLa3DFixedQueries
{
    private readonly int _slots, _width;
    private readonly Dense? _up, _down;
    private readonly Tensor? _learned;

    internal CoLa3DFixedQueries(IReadOnlyDictionary<string, Tensor> w, CoLa3DConfig config)
    {
        _slots = config.MaxParts;
        _width = config.HiddenDim;
        if (config.QueryInit == CoLa3DConfig.QueryInitKind.Learnable)
        {
            _learned = w["query.learnable"];
            return;
        }

        _up = LoadDense(w, "query.positional.up");
        _down = LoadDense(w, "query.positional.down");
    }

    /// <summary>Owned query rows <c>[1, slots, C]</c>.</summary>
    internal Tensor Compute(IBackend backend)
    {
        if (_learned is not null)
        {
            Tensor copy = Alloc(1, _slots, _width);
            _learned.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
            return copy;
        }

        // part_indices = arange(slots) / slots, embedded then mapped through Linear → SiLU → Linear.
        float[] indices = new float[_slots];
        for (int i = 0; i < _slots; i++)
            indices[i] = (float)i / _slots;
        Tensor embed = Alloc(1, _slots, PartFrequencyDim);
        CoLa3DOps.SinusoidalEmbedding(indices, PartFrequencyDim, embed.AsSpan<float>());

        Tensor hidden = Linear(backend, embed, _up!.Value, _width);
        embed.Dispose();
        Tensor act = new(hidden.Shape, DType.F32);
        backend.Silu(act, hidden);
        hidden.Dispose();
        Tensor output = Linear(backend, act, _down!.Value, _width);
        act.Dispose();
        return output;
    }
}
