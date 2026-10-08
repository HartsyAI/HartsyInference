using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Draft attention of one DSpark stage (upstream <c>DSparkAttention</c>): a window-only attention in which the block's queries see the sliding window of the
/// target's latents and the block's own latents, with no causal mask inside the block.</summary>
/// <remarks>Seeding writes the target's latents of the prompt into the window, as upstream's prefill branch does. A draft pass first writes the current target latent into
/// its window slot, then attends over the visible window and the block, as upstream's decode branch does.</remarks>
internal sealed class DeepSeekV41DSparkAttention
{
    private readonly IBackend _backend;
    private readonly DeepSeekV41AttentionSettings _settings;
    private readonly DeepSeekV41AttentionWeights _w;
    private readonly DeepSeekV41RopeTable _rope;

    public DeepSeekV41DSparkAttention(IBackend backend, DeepSeekV41AttentionSettings settings, DeepSeekV41AttentionWeights weights, DeepSeekV41RopeTable rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        if (settings.CompressRatio != 0) throw new ArgumentException("The DSpark draft attention is window-only.", nameof(settings));
        if (rope.HalfDim * 2 != settings.RopeDim) throw new ArgumentException("The rope table width must equal RopeDim.", nameof(rope));
        _backend = backend;
        _settings = settings;
        _w = weights;
        _rope = rope;
    }

    /// <summary>The attention settings this stage was built with, used to size its window.</summary>
    public DeepSeekV41AttentionSettings Settings => _settings;

    /// <summary>Writes the target's latents of positions <c>0 .. tokens-1</c> into the window; <paramref name="mainX"/> is <c>[tokens, Dim]</c>.</summary>
    public void Seed(ReadOnlySpan<float> mainX, int tokens, DeepSeekV41AttentionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        int hd = _settings.HeadDim, win = _settings.Window, dim = _settings.Dim;
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (mainX.Length != (long)tokens * dim) throw new ArgumentException("mainX must hold tokens x Dim values.", nameof(mainX));
        if (tokens > _rope.Length) throw new ArgumentOutOfRangeException(nameof(tokens), "The rope table does not reach these positions.");

        float[] kv = _w.Wkv.Linear(mainX, tokens, dim, hd);
        DeepSeekV41HostMath.RmsNormRows(kv, _w.KvNorm, hd, _settings.NormEps);
        DeepSeekV41Rotary.Rotate(_backend, _rope, _settings.RopeDim, kv, [1, tokens, hd], Range(0, tokens), hd - _settings.RopeDim, inverse: false);
        if (_settings.QuantizeLatents) RoundTrip(kv, hd);
        for (int p = Math.Max(0, tokens - win); p < tokens; p++)
            kv.AsSpan(p * hd, hd).CopyTo(state.Window.AsSpan((p % win) * hd, hd));
    }

    /// <summary>One draft pass: <paramref name="x"/> is the block's input <c>[block, Dim]</c> for positions <paramref name="startPos"/>+1 onward, and
    /// <paramref name="mainX"/> is the target's latent input <c>[Dim]</c> for position <paramref name="startPos"/>. Writes <c>[block, Dim]</c> to <paramref name="y"/>.</summary>
    public void Draft(ReadOnlySpan<float> x, int block, ReadOnlySpan<float> mainX, int startPos, DeepSeekV41AttentionState state, Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(state);
        int hd = _settings.HeadDim, rd = _settings.RopeDim, heads = _settings.Heads, win = _settings.Window, dim = _settings.Dim;
        if (block < 1) throw new ArgumentOutOfRangeException(nameof(block));
        if (x.Length != (long)block * dim || y.Length != x.Length) throw new ArgumentException("x and y must each hold block x Dim values.");
        if (mainX.Length != dim) throw new ArgumentException("mainX must hold Dim values.", nameof(mainX));
        if (startPos < 0 || startPos + block > _rope.Length) throw new ArgumentOutOfRangeException(nameof(startPos), "The rope table does not reach these positions.");

        // the current target latent enters the window first, so the block sees the position it continues from
        float[] mainKv = _w.Wkv.Linear(mainX, 1, dim, hd);
        DeepSeekV41HostMath.RmsNormRows(mainKv, _w.KvNorm, hd, _settings.NormEps);
        DeepSeekV41Rotary.Rotate(_backend, _rope, rd, mainKv, [1, 1, hd], [startPos], hd - rd, inverse: false);
        if (_settings.QuantizeLatents) RoundTrip(mainKv, hd);
        mainKv.CopyTo(state.Window.AsSpan((startPos % win) * hd, hd));

        int[] positions = Range(startPos + 1, block);
        float[] qr = _w.WqA.Linear(x, block, dim, _settings.QLoraRank);
        DeepSeekV41HostMath.RmsNormRows(qr, _w.QNorm, _settings.QLoraRank, _settings.NormEps);
        float[] q = _w.WqB.Linear(qr, block, _settings.QLoraRank, heads * hd);
        DeepSeekV41Rotary.Rotate(_backend, _rope, rd, q, [1, block, heads, hd], positions, hd - rd, inverse: false);

        float[] blockKv = _w.Wkv.Linear(x, block, dim, hd);
        DeepSeekV41HostMath.RmsNormRows(blockKv, _w.KvNorm, hd, _settings.NormEps);
        DeepSeekV41Rotary.Rotate(_backend, _rope, rd, blockKv, [1, block, hd], positions, hd - rd, inverse: false);
        if (_settings.QuantizeLatents) RoundTrip(blockKv, hd);

        // the window ring, then the block's latents, as upstream concatenates them; indices are the same for every query of the block
        float[] rows = new float[(win + block) * hd];
        state.Window.AsSpan().CopyTo(rows);
        blockKv.AsSpan().CopyTo(rows.AsSpan(win * hd));
        int visible = Math.Min(win, startPos + 1), k = visible + block;
        int[] indices = new int[block * k];
        for (int b = 0; b < block; b++)
        {
            for (int i = 0; i < visible; i++) indices[b * k + i] = i;
            for (int j = 0; j < block; j++) indices[b * k + visible + j] = win + j;
        }

        using Tensor query = DeepSeekV41HostMath.Tensor(q, block, heads, hd);
        using Tensor output = new(new TensorShape(block, heads, hd), DType.F32);
        using Tensor windowCodes = DeepSeekV41HostMath.Tensor(rows, win + block, hd);
        using Tensor indexT = DeepSeekV41HostMath.Tensor(indices, block, k);
        using Tensor sink = DeepSeekV41HostMath.Tensor(_w.Sink, heads);
        LatentSource window = new(LatentEncoding.F32, windowCodes, null, win + block, hd);
        _backend.SparseLatentAttention(output, query, window, LatentSource.Empty, indexT, win + block, sink, 1f / MathF.Sqrt(hd));

        float[] o = output.AsReadOnlySpan<float>().ToArray();
        DeepSeekV41Rotary.Rotate(_backend, _rope, rd, o, [1, block, heads, hd], positions, hd - rd, inverse: true);
        float[] grouped = new float[block * _settings.OGroups * _settings.OLoraRank];
        DeepSeekV41GroupedProjection.Apply(o, _w.WoA, block, _settings.OGroups, _settings.OLoraRank, _settings.GroupDim, grouped);
        _w.WoB.Linear(grouped, block, _settings.OGroups * _settings.OLoraRank, dim).CopyTo(y);
    }

    private void RoundTrip(float[] data, int width)
    {
        using Tensor t = DeepSeekV41HostMath.Tensor(data, data.Length / width, width);
        _backend.ActQuantDequantInPlace(t, LatentEncoding.Fp8E4M3Ue8m0x32);
        t.AsReadOnlySpan<float>().CopyTo(data);
    }

    private static int[] Range(int start, int count)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = start + i;
        return r;
    }
}
