using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Generates deterministic small-magnitude F32 weights for every parameter a module requests, keyed by
/// name so a given key always yields the same values across runs. Used to exercise the Grounding DINO forward pass
/// over synthetic weights (shape / plumbing verification) before any real checkpoint exists. Values are kept tiny
/// (~1e-2) so the untrained forward pass stays numerically bounded — no NaNs from a random deep transformer.</summary>
public sealed class SyntheticWeightBag : IGroundingWeightBag
{
    private readonly int _seed;
    private readonly List<Tensor> _created = new();

    public SyntheticWeightBag(int seed = 1234)
    {
        _seed = seed;
    }

    /// <summary>Every tensor handed out so far — lets the caller preload / dispose them.</summary>
    public IReadOnlyList<Tensor> Created => _created;

    public Tensor Get(string name, params int[] dims)
    {
        long count = 1;
        for (int i = 0; i < dims.Length; i++)
            count *= dims[i];
        Tensor tensor = new Tensor(new TensorShape(ToLong(dims)), DType.F32);
        Span<float> data = tensor.AsSpan<float>();

        // A cheap per-(name, index) hash → deterministic pseudo-random values in roughly [-0.05, 0.05]. A stable
        // string hash (not string.GetHashCode, which is randomized per process) keeps runs reproducible.
        uint nameHash = StableHash(name) ^ (uint)_seed;
        for (long i = 0; i < count; i++)
        {
            uint h = nameHash + (uint)(i * 2654435761u);
            h ^= h >> 15;
            h *= 2246822519u;
            h ^= h >> 13;
            float unit = (h & 0xFFFFFF) / (float)0xFFFFFF;
            data[(int)i] = (unit - 0.5f) * 0.1f;
        }
        _created.Add(tensor);
        return tensor;
    }

    private static uint StableHash(string s)
    {
        uint hash = 2166136261u;
        for (int i = 0; i < s.Length; i++)
        {
            hash ^= s[i];
            hash *= 16777619u;
        }
        return hash;
    }

    private static long[] ToLong(int[] dims)
    {
        long[] result = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++)
            result[i] = dims[i];
        return result;
    }
}
