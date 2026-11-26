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
        if (!ReferenceEquals(cache.Backend, backend))
        {
            throw new ArgumentException("The cache's segments live on its own backend; the forward must run on that same instance.", nameof(cache));
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
        using Tensor cos = DuoDeviceWindow.Upload(new TensorShape(seq, hd), cosValues);
        using Tensor sin = DuoDeviceWindow.Upload(new TensorShape(seq, hd), sinValues);

        Tensor stream = BuildDuoStream(backend, layout, refinerGroups, DuoAttentionMode.Causal, textStates, videoRows, audioRows,
            hidden, layout.TextRows.Length, layout.VideoRows.Length, layout.AudioRows.Length);
        // The rows this forward keeps for a commit, per block, as head-major device tensors the commit slices and then frees.
        Tensor?[] keptKeys = new Tensor?[_config.NumLayers];
        Tensor?[] keptValues = new Tensor?[_config.NumLayers];
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
                        Tensor attn = CachedAttention(backend, x, $"{p}.attn", cos, sin, cache.Store, block, segments, commit,
                            out Tensor? keys, out Tensor? values);
                        keptKeys[block] = keys;
                        keptValues[block] = values;
                        return attn;
                    }));
            }

            (Tensor video, Tensor? audio) = DuoHead(backend, stream, tEmb, layout, times.Length, audioRows is not null);
            if (commit)
            {
                try
                {
                    cache.Commit(keptKeys!, keptValues!, layout.Groups.Where(g => IsRetained(g.Kind)).ToList());
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
            foreach (Tensor? kept in keptKeys)
            {
                kept?.Dispose();
            }
            foreach (Tensor? kept in keptValues)
            {
                kept?.Dispose();
            }
        }
    }

    /// <summary>Attention over <c>[cache | current]</c>, one shared-dispatcher call per segment. Returns the output projection for the current
    /// rows. When <paramref name="keepRows"/> is true, also returns the current rows' keys and values as head-major device tensors
    /// <c>[1, heads, rows, head dim]</c>, keys already rotated; the caller owns them. Otherwise both are null and nothing is kept.</summary>
    private Tensor CachedAttention(IBackend backend, Tensor x, string prefix, Tensor cos, Tensor sin, KvSegmentStore store, int block,
        KeyRangeSegment[] segments, bool keepRows, out Tensor? keysNow, out Tensor? valuesNow)
    {
        keysNow = null;
        valuesNow = null;
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;

        using Tensor qkv = new Tensor(new TensorShape(seq, inner * 3), DType.F32);
        backend.Linear(qkv, x, Require($"{prefix}.qkv_proj.weight"), Optional($"{prefix}.qkv_proj.bias"));
        Tensor q = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor? kAll = null, vAll = null;
        bool handedOff = false;
        try
        {
            backend.QkvSplitNormHeadMajor(q, k, v, qkv, Require($"{prefix}.q_norm.weight"), Require($"{prefix}.k_norm.weight"),
                _config.QkNormEps);
            int rotary = MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen);
            backend.Rotary(RotaryCall.HeadMajor(q, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));
            backend.Rotary(RotaryCall.HeadMajor(k, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));

            // The history is read from the store: its segments joined along the token axis, then the current rows after them.
            // ProbeV is a diagnostic logger for the dense path and is not run here.
            Tensor keysAll = k, valuesAll = v;
            int count = store.Count(block);
            if (count > 0)
            {
                (Tensor pastKeys, Tensor pastValues) = store.Join(block, 0, count);
                try
                {
                    kAll = ConcatTokens(backend, pastKeys, k);
                    vAll = ConcatTokens(backend, pastValues, v);
                }
                finally
                {
                    pastKeys.Dispose();
                    pastValues.Dispose();
                }
                keysAll = kAll;
                valuesAll = vAll;
            }
            Tensor scores = SegmentedAttention.RunHeadMajor(backend, _attention, q, keysAll, valuesAll, segments);
            try
            {
                using Tensor merged = new Tensor(new TensorShape(seq, inner), DType.F32);
                backend.Permute0213(merged, scores, heads, seq, hd);
                Tensor outT = new Tensor(x.Shape, BodyDType);
                backend.Linear(outT, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
                if (keepRows)
                {
                    keysNow = k;
                    valuesNow = v;
                    handedOff = true;
                }
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
            kAll?.Dispose();
            vAll?.Dispose();
            if (!handedOff)
            {
                k.Dispose();
                v.Dispose();
            }
        }
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

    /// <summary>Concatenates two head-major <c>[1, heads, rows, head dim]</c> tensors along the token axis into a new tensor the caller owns.</summary>
    private static Tensor ConcatTokens(IBackend backend, Tensor first, Tensor second)
    {
        Span<long> dims = stackalloc long[first.Shape.Rank];
        first.Shape.CopyDimsTo(dims);
        dims[2] += second.Shape[2];
        Tensor joined = new(new TensorShape(dims), first.DType);
        try
        {
            backend.Concat(joined, [first, second], 2);
        }
        catch
        {
            joined.Dispose();
            throw;
        }
        return joined;
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
