using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The cached Duo forward (PR 2h): a denoise that reads a committed history and, when asked, commits its own clean rows.
/// Attention keys are <c>[cache | current]</c>, so the bias is <c>[S, past + S]</c>. Each block projects the current rows, rotates
/// their keys, and concatenates them after the cached keys. Only the dense route is implemented; a layout past
/// <see cref="MaxDuoDenseBiasRows"/> in either direction is refused.</summary>
public sealed unsafe partial class MiniMaxH3Transformer
{
    /// <summary>Runs one causal Duo denoise over <paramref name="layout"/>, reading <paramref name="cache"/>. When <paramref name="commit"/>
    /// is true, the reference, shared and clean rows of this forward are appended to the cache after the outputs are computed. An uncommitted
    /// forward leaves the cache unchanged.</summary>
    /// <param name="textStates">Text conditioning rows before <c>context_embedder</c>, or null when the layout has no text rows.</param>
    /// <exception cref="NotSupportedException">A bidirectional mode, a tiled-size layout, or a route this path does not implement.</exception>
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
        if (BodyDType != DType.F32)
        {
            throw new NotSupportedException("The Duo forward runs an F32 body; its zero-row gather is F32-only.");
        }
        if (layout.VideoRows.Length == 0)
        {
            throw new NotSupportedException("A Duo layout with no video rows is not supported; every target carries video.");
        }
        if (audioRows is null && layout.AudioRows.Length > 0)
        {
            throw new ArgumentException("The layout has audio rows, so audioRows must be given.", nameof(audioRows));
        }
        if (audioRows is not null && !_config.HasAudio)
        {
            throw new ArgumentException("This checkpoint has no audio stream, so audioRows must be null.", nameof(audioRows));
        }
        if (videoRows.Shape[0] != layout.VideoRows.Length)
        {
            throw new ArgumentException($"Video rows {videoRows.Shape[0]} do not match the layout's {layout.VideoRows.Length}.", nameof(videoRows));
        }
        if (audioRows is not null && audioRows.Shape[0] != layout.AudioRows.Length)
        {
            throw new ArgumentException($"Audio rows {audioRows.Shape[0]} do not match the layout's {layout.AudioRows.Length}.", nameof(audioRows));
        }
        if (times.Length == 0)
        {
            throw new ArgumentException("At least one timestep is needed.", nameof(times));
        }
        int textRows = textStates is null ? 0 : (int)textStates.Shape[0];
        if (textRows == 0 && layout.TextRows.Length > 0)
        {
            throw new ArgumentException("The layout has text rows, so textStates must be given.", nameof(textStates));
        }
        ValidateDuoRefinerGroups(refinerGroups, textRows);
        ValidateDuoTimeIds(layout, times.Length);

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
                $"The cached Duo forward is dense: {past} cached rows and {seq} current rows exceed {MaxDuoDenseBiasRows}. "
                + "Tiled cached attention is not implemented.");
        }

        // The keys are the cache groups, then this forward's groups shifted past the cache.
        List<DuoGroup> keyGroups = [.. cache.Groups];
        foreach (DuoGroup g in layout.Groups)
        {
            keyGroups.Add(g with { Start = g.Start + past, End = g.End + past });
        }
        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build(layout.Groups, keyGroups, DuoAttentionMode.Causal);
        DuoAttentionBias.ValidateRows(rects, seq, past + seq);
        using Tensor bias = HostTensor(new TensorShape(seq, past + seq), DuoAttentionBias.BuildRows(rects, seq, 0, seq, past + seq));

        using Tensor tEmb = BuildTimeEmbedding(backend, times);
        using Tensor modIndex = BuildDuoModulationIndex(layout);
        (float[] cosValues, float[] sinValues) = DuoRope.Tables(layout.Positions, hd);
        using Tensor cos = HostTensor(new TensorShape(seq, hd), cosValues);
        using Tensor sin = HostTensor(new TensorShape(seq, hd), sinValues);

        Tensor stream = BuildDuoStreamCached(backend, layout, refinerGroups, textStates, textRows, videoRows, audioRows, hidden);
        float[][] keptKeys = new float[_config.NumLayers][];
        float[][] keptValues = new float[_config.NumLayers][];
        try
        {
            for (int i = 0; i < _config.NumLayers; i++)
            {
                string p = $"blocks.{i}";
                Tensor[] mod = Adaln(backend, tEmb, $"{p}.adaln_proj", expand: 6, modalities: 3);
                try
                {
                    Tensor normed = Norm(backend, stream, $"{p}.norm1.weight", _config.NormEps, BodyDType);
                    Tensor modulated = Modulate(backend, normed, mod[0], mod[1], modIndex);
                    normed.Dispose();
                    Tensor attn = CachedAttention(backend, modulated, $"{p}.attn", cos, sin, cache.Layer(i), bias,
                        out float[] keysNow, out float[] valuesNow);
                    modulated.Dispose();
                    keptKeys[i] = keysNow;
                    keptValues[i] = valuesNow;
                    Tensor gatedAttn = Gate(backend, stream, attn, mod[2], modIndex);
                    attn.Dispose();
                    Swap(ref stream, gatedAttn);

                    Tensor normed2 = Norm(backend, stream, $"{p}.norm2.weight", _config.NormEps, BodyDType);
                    Tensor modulated2 = Modulate(backend, normed2, mod[3], mod[4], modIndex);
                    normed2.Dispose();
                    Tensor mlp = Mlp(backend, modulated2, $"{p}.mlp");
                    modulated2.Dispose();
                    Tensor gatedMlp = Gate(backend, stream, mlp, mod[5], modIndex);
                    mlp.Dispose();
                    Swap(ref stream, gatedMlp);
                }
                finally
                {
                    foreach (Tensor t in mod) t.Dispose();
                }
            }

            (Tensor video, Tensor? audio) = DuoHead(backend, stream, tEmb, layout, times.Length, audioRows is not null);
            if (commit)
            {
                CommitDuoRows(layout, cache, keptKeys, keptValues, heads, hd, seq);
            }
            return (video, audio);
        }
        finally
        {
            stream.Dispose();
        }
    }

    /// <summary>Attention over <c>[cache | current]</c>. Returns the output projection for the current rows, and the current rows' keys
    /// and values as head-major <c>[heads, rows, hd]</c> host arrays (keys already rotated), for a commit.</summary>
    private Tensor CachedAttention(IBackend backend, Tensor x, string prefix, Tensor cos, Tensor sin, DuoKvCache past,
        Tensor bias, out float[] keysNow, out float[] valuesNow)
    {
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;
        using Tensor qkv = new Tensor(new TensorShape(seq, inner * 3), DType.F32);
        backend.Linear(qkv, x, Require($"{prefix}.qkv_proj.weight"), Optional($"{prefix}.qkv_proj.bias"));
        Tensor q = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
        try
        {
            backend.QkvSplitNormHeadMajor(q, k, v, qkv, Require($"{prefix}.q_norm.weight"), Require($"{prefix}.k_norm.weight"),
                _config.QkNormEps);
            int rotary = MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen);
            backend.Rotary(RotaryCall.HeadMajor(q, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));
            backend.Rotary(RotaryCall.HeadMajor(k, cos, sin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotary));
            keysNow = k.AsSpan<float>().ToArray();
            valuesNow = v.AsSpan<float>().ToArray();

            int rows = past.Rows;
            using Tensor kAll = HostTensor(new TensorShape(1, heads, rows + seq, hd),
                HeadMajorWithPast(past.Keys, rows, keysNow, seq, heads, hd));
            using Tensor vAll = HostTensor(new TensorShape(1, heads, rows + seq, hd),
                HeadMajorWithPast(past.Values, rows, valuesNow, seq, heads, hd));
            using Tensor scores = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            backend.ScaledDotProductAttention(scores, q, kAll, vAll, bias, 1f / MathF.Sqrt(hd));
            using Tensor merged = new Tensor(new TensorShape(seq, inner), DType.F32);
            backend.Permute0213(merged, scores, heads, seq, hd);
            Tensor outT = new Tensor(x.Shape, BodyDType);
            backend.Linear(outT, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
            return outT;
        }
        finally
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
        }
    }

    /// <summary>Head-major <c>[heads, pastRows + curRows, hd]</c> from the cache's row-major past and the current head-major rows.</summary>
    private static float[] HeadMajorWithPast(ReadOnlySpan<float> pastRowMajor, int pastRows, float[] curHeadMajor, int curRows,
        int heads, int hd)
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
                curHeadMajor.AsSpan((h * curRows + t) * hd, hd).CopyTo(all.AsSpan((h * total + pastRows + t) * hd, hd));
            }
        }
        return all;
    }

    /// <summary>Appends the reference, shared and clean rows of this forward to the cache, in group order, as upstream's retain does.</summary>
    private static void CommitDuoRows(DuoLayout layout, DuoStreamCache cache, float[][] keysHeadMajor, float[][] valuesHeadMajor,
        int heads, int hd, int seq)
    {
        int inner = heads * hd;
        List<int> retained = [];
        List<DuoGroup> newGroups = [];
        int cursor = cache.Length;
        foreach (DuoGroup g in layout.Groups)
        {
            if (g.Kind is not (DuoGroupKind.Reference or DuoGroupKind.Shared or DuoGroupKind.Clean))
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

        int rows = retained.Count;
        float[][] keysRowMajor = new float[cache.Blocks][];
        float[][] valuesRowMajor = new float[cache.Blocks][];
        for (int b = 0; b < cache.Blocks; b++)
        {
            keysRowMajor[b] = ToRowMajor(keysHeadMajor[b], retained, seq, heads, hd, inner);
            valuesRowMajor[b] = ToRowMajor(valuesHeadMajor[b], retained, seq, heads, hd, inner);
        }
        cache.Commit(keysRowMajor, valuesRowMajor, rows, newGroups);
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

    /// <summary>The packed stream for a cached forward: refined text rows when present, projected video and audio, and a zero row for padding.</summary>
    private Tensor BuildDuoStreamCached(IBackend backend, DuoLayout layout, IReadOnlyList<DuoGroup> refinerGroups,
        Tensor? textStates, int textRows, Tensor videoRows, Tensor? audioRows, int hidden)
    {
        int textCount = layout.TextRows.Length, videoCount = layout.VideoRows.Length, audioCount = layout.AudioRows.Length;
        using Tensor? refined = textRows > 0 ? RefineDuoText(backend, textStates!, refinerGroups, DuoAttentionMode.Causal) : null;
        using Tensor? selected = textCount > 0 ? Gather(backend, refined!, layout.TextSourceRows) : null;
        using Tensor videoEmbed = ProjectIntoStream(backend, videoRows, "video_patch_proj", hidden);
        using Tensor? audioEmbed = audioCount > 0 ? ProjectIntoStream(backend, audioRows!, "audio_patch_proj", hidden) : null;
        using Tensor zero = new Tensor(new TensorShape(1, hidden), DType.F32);
        zero.AsSpan<float>().Clear();

        List<Tensor> pieces = new List<Tensor>(4);
        if (selected is not null) pieces.Add(selected);
        pieces.Add(videoEmbed);
        if (audioEmbed is not null) pieces.Add(audioEmbed);
        pieces.Add(zero);

        int sourceRows = textCount + videoCount + audioCount + 1;
        using Tensor source = new Tensor(new TensorShape(sourceRows, hidden), DType.F32);
        backend.Concat(source, CollectionsMarshal.AsSpan(pieces), 0);

        int zeroRow = sourceRows - 1;
        int[] packed = new int[layout.Length];
        Array.Fill(packed, zeroRow);
        for (int k = 0; k < textCount; k++) packed[layout.TextRows[k]] = k;
        for (int k = 0; k < videoCount; k++) packed[layout.VideoRows[k]] = textCount + k;
        for (int k = 0; k < audioCount; k++) packed[layout.AudioRows[k]] = textCount + videoCount + k;
        return Gather(backend, source, packed);
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
