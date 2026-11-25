using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The cached Duo forward (PR 2h): a denoise that reads a committed history and, when asked, commits its own clean rows.
/// Attention keys are <c>[cache | current]</c> and each query group attends the key rows it may read, so there is no bias. Each block runs
/// the shared H3 block body, with the attention step supplied as a delegate. The attention is not chunked; a layout past
/// <see cref="MaxDuoDenseBiasRows"/> in either direction is refused.</summary>
public sealed unsafe partial class MiniMaxH3Transformer
{
    /// <summary>Runs one causal Duo denoise over <paramref name="layout"/>, reading <paramref name="cache"/>. When <paramref name="commit"/>
    /// is true, the reference, shared and clean rows of this forward are appended to the cache after the outputs are computed, all or nothing.
    /// An uncommitted forward leaves the cache unchanged.</summary>
    /// <param name="textStates">Text conditioning rows before <c>context_embedder</c>, or null when the layout has no text rows.</param>
    /// <exception cref="NotSupportedException">A bidirectional mode, a layout past the dense limit, or a route this path does not implement.</exception>
    /// <exception cref="InvalidOperationException">A commit would exceed the cache capacity. Refused before any allocation.</exception>
    internal (Tensor Video, Tensor? Audio) ForwardDuoCached(IBackend backend, DuoLayout layout, DuoAttentionMode mode,
        IReadOnlyList<DuoGroup> refinerGroups, Tensor? textStates, Tensor videoRows, Tensor? audioRows,
        float[] times, DuoStreamCache cache, bool commit)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(refinerGroups);
        ArgumentNullException.ThrowIfNull(videoRows);
        ArgumentNullException.ThrowIfNull(times);
        ArgumentNullException.ThrowIfNull(cache);
        if (mode != DuoAttentionMode.Causal)
        {
            throw new NotSupportedException("A cached Duo forward is causal only; a bidirectional score never reads the committed history.");
        }
        ValidateDuoInputs(layout, refinerGroups, textStates is null ? 0 : (int)textStates.Shape[0], videoRows, audioRows, times.Length);

        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd, hidden = _config.HiddenSize;
        if (cache.Blocks != _config.NumLayers || cache.Width != inner)
        {
            throw new ArgumentException(
                $"The cache holds {cache.Blocks} blocks of width {cache.Width}; this model needs {_config.NumLayers} of width {inner}.", nameof(cache));
        }
        int seq = layout.Length, past = cache.Length;
        if (seq > MaxDuoDenseBiasRows || past + seq > MaxDuoDenseBiasRows)
        {
            throw new NotSupportedException(
                $"The cached Duo forward is not chunked: {past} cached rows and {seq} current rows exceed {MaxDuoDenseBiasRows}. "
                + "Tiled cached attention is not implemented.");
        }
        // Refuse an overflowing commit before anything is allocated, so a refused commit costs nothing.
        int retained = RetainedRows(layout);
        if (commit && retained > cache.CapacityRows - past)
        {
            throw new InvalidOperationException(
                $"Committing {retained} rows to {past} would exceed the {cache.CapacityRows}-row cache capacity.");
        }

        // The keys are the cache groups, then this forward's groups shifted past the cache.
        List<DuoGroup> keyGroups = [.. cache.Groups];
        foreach (DuoGroup g in layout.Groups)
        {
            keyGroups.Add(g with { Start = g.Start + past, End = g.End + past });
        }
        KeyRangeSegment[] segments = DuoMask.Segments(DuoMask.Build(layout.Groups, keyGroups, DuoAttentionMode.Causal), seq, past + seq);

        using Tensor tEmb = BuildTimeEmbedding(backend, times);
        using Tensor modIndex = BuildDuoModulationIndex(layout);
        (float[] cosValues, float[] sinValues) = DuoRope.Tables(layout.Positions, hd);
        using Tensor cos = HostTensor(new TensorShape(seq, hd), cosValues);
        using Tensor sin = HostTensor(new TensorShape(seq, hd), sinValues);

        Tensor stream = BuildDuoStream(backend, layout, refinerGroups, DuoAttentionMode.Causal, textStates, videoRows, audioRows,
            hidden, layout.TextRows.Length, layout.VideoRows.Length, layout.AudioRows.Length);
        float[][] keptKeys = new float[_config.NumLayers][];
        float[][] keptValues = new float[_config.NumLayers][];
        try
        {
            for (int i = 0; i < _config.NumLayers; i++)
            {
                int block = i;
                string p = $"blocks.{i}";
                // The block body is the shared H3 one. Only the attention step differs: it reads the cache and keeps this block's rows
                // for a commit. No chunking applies here, so the block runs unchunked and the delegate is the whole attention.
                ForwardNamedBlock(backend, ref stream, tEmb, modIndex, cos, sin, p, int.MaxValue,
                    BlockAttention.Override(x =>
                    {
                        Tensor attn = CachedAttention(backend, x, $"{p}.attn", cos, sin, cache.Layer(block), segments, commit,
                            out float[]? keys, out float[]? values);
                        keptKeys[block] = keys!;
                        keptValues[block] = values!;
                        return attn;
                    }));
            }

            (Tensor video, Tensor? audio) = DuoHead(backend, stream, tEmb, layout, times.Length, audioRows is not null);
            if (commit)
            {
                try
                {
                    CommitDuoRows(layout, cache, keptKeys, keptValues, heads, hd, seq);
                }
                catch
                {
                    video.Dispose();
                    audio?.Dispose();
                    throw;
                }
            }
            return (video, audio);
        }
        finally
        {
            stream.Dispose();
        }
    }

    /// <summary>Attention over <c>[cache | current]</c>, one shared-dispatcher call per segment. Returns the output projection for the current
    /// rows. When <paramref name="keepRows"/> is true, also returns the current rows' keys and values as head-major <c>[heads, rows, hd]</c>
    /// host arrays, keys already rotated, for a commit; otherwise both are null and nothing is copied.</summary>
    private Tensor CachedAttention(IBackend backend, Tensor x, string prefix, Tensor cos, Tensor sin, DuoKvCache past,
        KeyRangeSegment[] segments, bool keepRows, out float[]? keysNow, out float[]? valuesNow)
    {
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;
        int rows = past.Rows;

        using Tensor qkv = new Tensor(new TensorShape(seq, inner * 3), DType.F32);
        backend.Linear(qkv, x, Require($"{prefix}.qkv_proj.weight"), Optional($"{prefix}.qkv_proj.bias"));
        Tensor q = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor? kAll = null, vAll = null;
        try
        {
            backend.QkvSplitNormHeadMajor(q, k, v, qkv, Require($"{prefix}.q_norm.weight"), Require($"{prefix}.k_norm.weight"),
                _config.QkNormEps);
            int rotary = MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen);
            backend.Rotary(RotaryCall.HeadMajor(q, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));
            backend.Rotary(RotaryCall.HeadMajor(k, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));
            if (keepRows)
            {
                keysNow = k.AsSpan<float>().ToArray();
                valuesNow = v.AsSpan<float>().ToArray();
            }
            else
            {
                keysNow = null;
                valuesNow = null;
            }

            // ProbeV is a diagnostic logger for the dense path and is not run here.
            kAll = HostTensor(new TensorShape(1, heads, rows + seq, hd),
                HeadMajorWithPast(past.Keys, rows, k.AsSpan<float>(), seq, heads, hd));
            vAll = HostTensor(new TensorShape(1, heads, rows + seq, hd),
                HeadMajorWithPast(past.Values, rows, v.AsSpan<float>(), seq, heads, hd));
            Tensor scores = SegmentedAttention.RunHeadMajor(backend, _attention, q, kAll, vAll, segments);
            try
            {
                using Tensor merged = new Tensor(new TensorShape(seq, inner), DType.F32);
                backend.Permute0213(merged, scores, heads, seq, hd);
                Tensor outT = new Tensor(x.Shape, BodyDType);
                backend.Linear(outT, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
                return outT;
            }
            finally
            {
                scores.Dispose();
            }
        }
        finally
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            kAll?.Dispose();
            vAll?.Dispose();
        }
    }

    /// <summary>Head-major <c>[heads, pastRows + curRows, hd]</c> from the cache's row-major past and the current head-major rows.</summary>
    private static float[] HeadMajorWithPast(ReadOnlySpan<float> pastRowMajor, int pastRows, ReadOnlySpan<float> curHeadMajor,
        int curRows, int heads, int hd)
    {
        int inner = heads * hd, total = pastRows + curRows;
        float[] all = new float[heads * total * hd];
        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < pastRows; t++)
            {
                pastRowMajor.Slice(t * inner + h * hd, hd).CopyTo(all.AsSpan((h * total + t) * hd, hd));
            }
            for (int t = 0; t < curRows; t++)
            {
                curHeadMajor.Slice((h * curRows + t) * hd, hd).CopyTo(all.AsSpan((h * total + pastRows + t) * hd, hd));
            }
        }
        return all;
    }

    /// <summary>Rows of this forward that a commit keeps: the reference, shared and clean groups, as upstream's retain selects them.</summary>
    private static int RetainedRows(DuoLayout layout)
    {
        int rows = 0;
        foreach (DuoGroup g in layout.Groups)
        {
            if (IsRetained(g.Kind))
            {
                rows += g.End - g.Start;
            }
        }
        return rows;
    }

    private static bool IsRetained(DuoGroupKind kind) => kind is DuoGroupKind.Reference or DuoGroupKind.Shared or DuoGroupKind.Clean;

    /// <summary>Appends the retained rows of this forward to the cache, in group order. Builds every array before the cache is touched.</summary>
    private static void CommitDuoRows(DuoLayout layout, DuoStreamCache cache, float[][] keysHeadMajor, float[][] valuesHeadMajor,
        int heads, int hd, int seq)
    {
        int inner = heads * hd;
        List<int> retained = [];
        List<DuoGroup> newGroups = [];
        int cursor = cache.Length;
        foreach (DuoGroup g in layout.Groups)
        {
            if (!IsRetained(g.Kind))
            {
                continue;
            }
            int n = g.End - g.Start;
            for (int r = g.Start; r < g.End; r++)
            {
                retained.Add(r);
            }
            newGroups.Add(g with { Start = cursor, End = cursor + n });
            cursor += n;
        }

        float[][] keysRowMajor = new float[cache.Blocks][];
        float[][] valuesRowMajor = new float[cache.Blocks][];
        for (int b = 0; b < cache.Blocks; b++)
        {
            keysRowMajor[b] = ToRowMajor(keysHeadMajor[b], retained, seq, heads, hd, inner);
            valuesRowMajor[b] = ToRowMajor(valuesHeadMajor[b], retained, seq, heads, hd, inner);
        }
        cache.Commit(keysRowMajor, valuesRowMajor, retained.Count, newGroups);
    }

    /// <summary>Picks rows from head-major <c>[heads, seq, hd]</c> and returns them row-major <c>[rows, heads * hd]</c>.</summary>
    private static float[] ToRowMajor(float[] headMajor, IReadOnlyList<int> rows, int seq, int heads, int hd, int inner)
    {
        float[] outRows = new float[rows.Count * inner];
        for (int j = 0; j < rows.Count; j++)
        {
            for (int h = 0; h < heads; h++)
            {
                headMajor.AsSpan((h * seq + rows[j]) * hd, hd).CopyTo(outRows.AsSpan(j * inner + h * hd, hd));
            }
        }
        return outRows;
    }

    /// <summary>norm_out and the two output heads over the packed stream. Shared by the cached and the uncached forward.</summary>
    private (Tensor Video, Tensor? Audio) DuoHead(IBackend backend, Tensor stream, Tensor tEmb, DuoLayout layout, int timestepCount,
        bool hasAudio)
    {
        Tensor normed = Norm(backend, stream, "final_layer.norm.weight", _config.FinalNormEps, DType.F32);
        try
        {
            Tensor[] headMod = Adaln(backend, tEmb, "final_layer.adaln_proj", expand: 2, modalities: 1);
            try
            {
                using Tensor timeIndex = BuildDuoTimeIndex(layout);
                using Tensor modulated = Modulate(backend, normed, headMod[0], headMod[1], timeIndex);
                Tensor video = LinearDuoHead(backend, Gather(backend, modulated, layout.VideoRows), "final_layer.video_out");
                try
                {
                    Tensor? audio = hasAudio && layout.AudioRows.Length > 0
                        ? LinearDuoHead(backend, Gather(backend, modulated, layout.AudioRows), "final_layer.audio_out")
                        : null;
                    return (video, audio);
                }
                catch
                {
                    video.Dispose();
                    throw;
                }
            }
            finally
            {
                foreach (Tensor t in headMod) t.Dispose();
            }
        }
        finally
        {
            normed.Dispose();
        }
    }
}
