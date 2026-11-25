namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The committed history of a Duo denoise: one <see cref="DuoKvCache"/> per block, and the groups that own its rows,
/// in cache coordinates. Keys are post-QK-norm and post-RoPE; values are as the V projection produces them. A commit appends only
/// the rows of reference, shared and clean groups, as upstream <c>retain</c> does. The cache is mutated only by a committing forward.
/// <b>CPU reference storage.</b></summary>
public sealed class DuoStreamCache
{
    private readonly DuoKvCache[] _layers;
    private readonly List<DuoGroup> _groups = [];

    /// <param name="blocks">Transformer blocks, one KV cache each.</param>
    /// <param name="width">Row width of each cache: heads times head dimension.</param>
    /// <param name="capacityRows">Rows each block's cache can hold. A commit beyond it throws.</param>
    public DuoStreamCache(int blocks, int width, int capacityRows)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(blocks);
        _layers = new DuoKvCache[blocks];
        for (int i = 0; i < blocks; i++)
        {
            _layers[i] = new DuoKvCache(capacityRows, width);
        }
        Width = width;
    }

    /// <summary>Transformer blocks this cache serves.</summary>
    public int Blocks => _layers.Length;

    /// <summary>Row width of each block's cache.</summary>
    public int Width { get; }

    /// <summary>Committed rows. Every block holds the same count.</summary>
    public int Length => _layers.Length == 0 ? 0 : _layers[0].Rows;

    /// <summary>The groups that own the committed rows, in cache coordinates.</summary>
    public IReadOnlyList<DuoGroup> Groups => _groups;

    /// <summary>The KV rows of one block.</summary>
    public DuoKvCache Layer(int block) => _layers[block];

    /// <summary>Appends one committed chunk: per block, <paramref name="keys"/> and <paramref name="values"/> hold <paramref name="rows"/> rows
    /// of <see cref="Width"/> floats, row-major. <paramref name="newGroups"/> name those rows in cache coordinates.</summary>
    internal void Commit(float[][] keys, float[][] values, int rows, IReadOnlyList<DuoGroup> newGroups)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(newGroups);
        if (rows > 0)
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].Append(keys[i], values[i]);
            }
        }
        _groups.AddRange(newGroups);
    }
}
