using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The L-RoPE tables one block's audio cross-attention rotates with: the queries' rows are every noisy token in frame order, the keys'
/// rows every noisy frame's audio tokens. Borrowed.</summary>
internal readonly record struct LongCatSpeakerRotation(Tensor QueryCos, Tensor QuerySin, Tensor KeyCos, Tensor KeySin);

/// <summary>What every block of one multi-speaker forward shares: the masks laid out as attention values, the weights that average the attention
/// map over the heads, and the audio keys' rotation tables. Each block computes its own reference attention map
/// (<c>get_attn_map_with_target</c>, meituan-longcat/LongCat-Video @ 58eb336) and from it the queries' tables.</summary>
/// <remarks>
/// The reference's map is, per head, the softmax of the noisy queries over the first latent frame's keys, summed inside each region's mask and
/// divided by the region's token count, then averaged over two groups of half the heads. The sum inside a mask is an attention whose values are the
/// masks, so it runs as the layer's ordinary attention with <c>V[head, key, r] = mask_r[key]</c> (the other channels zero), in query tiles, and one
/// matrix product then averages channel <c>r</c> over the heads. No score matrix is held. A region with no token gives zero: the pinned code divides
/// by zero there, and upstream pull request 159 clamps the divisor to one.
/// </remarks>
internal sealed class LongCatSpeakerPass : IDisposable
{
    private readonly int _heads, _headDim, _tileRows;
    private readonly int _regions, _referenceTokens;
    private readonly Tensor _values, _average, _keyCos, _keySin;
    private int _disposed;

    /// <summary>Builds the shared tensors for <paramref name="noisyFrames"/> noisy latent frames.</summary>
    public LongCatSpeakerPass(LongCatSpeakers speakers, int heads, int headDim, int contextTokens, int noisyFrames,
        int tileRows = LongCatBlock.DefaultTileRows)
    {
        ArgumentNullException.ThrowIfNull(speakers);
        ArgumentOutOfRangeException.ThrowIfLessThan(noisyFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(tileRows, 1);
        if (heads < 2)
            throw new ArgumentOutOfRangeException(nameof(heads), heads, "The reference averages two head groups, so it needs at least two heads.");
        if (speakers.Regions > headDim)
            throw new ArgumentException($"The reference map takes at most {headDim} regions (one per element of a head), "
                + $"got {speakers.Regions}.", nameof(speakers));
        Speakers = speakers;
        _heads = heads;
        _headDim = headDim;
        _tileRows = tileRows;
        _regions = speakers.Regions;
        _referenceTokens = speakers.GridHeight * speakers.GridWidth;
        ContextTokens = contextTokens;

        ReadOnlySpan<float> masks = speakers.Masks;
        _values = new Tensor(new TensorShape(1, heads, _referenceTokens, headDim), DType.F32);
        Span<float> values = _values.AsSpan<float>();
        values.Clear();
        for (int h = 0; h < heads; h++)
            for (int key = 0; key < _referenceTokens; key++)
                for (int r = 0; r < _regions; r++)
                    values[(h * _referenceTokens + key) * headDim + r] = masks[r * _referenceTokens + key];

        // Two groups of heads/2 heads, each averaged and then averaged together: the mean over the first 2·(heads/2) heads.
        int used = 2 * (heads / 2);
        _average = new Tensor(new TensorShape(_regions, (long)heads * headDim), DType.F32);
        Span<float> average = _average.AsSpan<float>();
        average.Clear();
        for (int r = 0; r < _regions; r++)
        {
            float count = 0f;
            for (int key = 0; key < _referenceTokens; key++)
                count += masks[r * _referenceTokens + key];
            float weight = 1f / (used * MathF.Max(count, 1f));
            for (int h = 0; h < used; h++)
                average[r * heads * headDim + h * headDim + r] = weight;
        }

        float[] frame = LongCatSpeakerRope.KeyPositions(speakers.Tracks, contextTokens, _regions);
        float[] positions = new float[noisyFrames * frame.Length];
        for (int f = 0; f < noisyFrames; f++)
            frame.CopyTo(positions, f * frame.Length);
        KeyPositions = positions;
        (_keyCos, _keySin) = LongCatSpeakerRope.Tables(positions, headDim);
    }

    /// <summary>The conditioning the pass was built from.</summary>
    public LongCatSpeakers Speakers { get; }

    /// <summary>Audio tokens per track and frame.</summary>
    public int ContextTokens { get; }

    /// <summary>The audio keys' positions, one per audio token of the noisy frames.</summary>
    public float[] KeyPositions { get; }

    /// <summary>The reference attention map <c>[regions, rows]</c> of the query rows from <paramref name="firstRow"/> on.
    /// <paramref name="queries"/> are the block's token-major queries after the QK norm and the 3D rotation; <paramref name="referenceKeys"/>
    /// are the first latent frame's keys after the same, head-major <c>[1, heads, tokens, headDim]</c>. Both stay the caller's. Reads the result to
    /// the host.</summary>
    public float[] ReferenceMap(IBackend backend, AttentionLayer layer, Tensor queries, int firstRow, Tensor referenceKeys)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int dim = _heads * _headDim;
        int rows = checked((int)(queries.ElementCount / dim) - firstRow);
        if (rows < 1)
            throw new ArgumentOutOfRangeException(nameof(firstRow), firstRow, "No query row is left for the reference map.");
        if (referenceKeys.Shape.Rank != 4 || referenceKeys.Shape[2] != _referenceTokens)
            throw new ArgumentException($"The reference keys must be [1, {_heads}, {_referenceTokens}, {_headDim}], got "
                + $"{referenceKeys.Shape}.", nameof(referenceKeys));
        float[] map = new float[_regions * rows];
        for (int start = 0; start < rows; start += _tileRows)
        {
            int count = Math.Min(_tileRows, rows - start);
            Tensor headMajor = new(new TensorShape(1, _heads, count, _headDim), DType.F32);
            try
            {
                using Tensor tile = new(new TensorShape(count, dim), DType.F32);
                backend.SliceRows(tile, queries, firstRow + start);
                backend.Permute0213(headMajor, tile, count, _heads, _headDim);
            }
            catch
            {
                headMajor.Dispose();
                throw;
            }
            // The layer takes the query over; the keys and the region values are borrowed.
            Tensor attended = layer.AttendHeadMajorOperands(backend, headMajor, referenceKeys, _values, default, OperandOwnership.Query);
            using Tensor flat = layer.MergeHeads(backend, attended, new TensorShape(count, dim));
            using Tensor mean = new(new TensorShape(count, _regions), DType.F32);
            backend.Linear(mean, flat, _average, null);
            ReadOnlySpan<float> host = mean.AsReadOnlySpan<float>();
            for (int t = 0; t < count; t++)
                for (int r = 0; r < _regions; r++)
                    map[r * rows + start + t] = host[t * _regions + r];
        }
        return map;
    }

    /// <summary>The rotation of one block from its queries' positions: new query tables (the caller owns and disposes them) beside the pass's key
    /// tables.</summary>
    public LongCatSpeakerRotation Rotation(float[] queryPositions)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        (Tensor cos, Tensor sin) = LongCatSpeakerRope.Tables(queryPositions, _headDim);
        return new LongCatSpeakerRotation(cos, sin, _keyCos, _keySin);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _values.Dispose();
        _average.Dispose();
        _keyCos.Dispose();
        _keySin.Dispose();
    }
}
