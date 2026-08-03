using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MiniMax-H3 DiT: a single-stream packed-token transformer that denoises video and stereo audio latents
/// jointly. Rows are laid out by <see cref="MiniMaxH3PackedLayout"/> as [text | conditioning | audio | video]; each
/// segment carries its own modulation row, selected by (timestep class, modality tag).</summary>
public sealed unsafe class MiniMaxH3Transformer : IDisposable
{
    private readonly MiniMaxH3Config _config;
    private readonly Dictionary<string, Tensor> _weights = new Dictionary<string, Tensor>();
    private bool _disposed;

    /// <summary>Modality tag per segment kind; adaln packs three modalities (video 0, text 1, audio 2) per row.</summary>
    private static int ModalityTag(MiniMaxH3SegmentKind kind) => kind switch
    {
        MiniMaxH3SegmentKind.Text => 1,
        MiniMaxH3SegmentKind.Audio or MiniMaxH3SegmentKind.RefAudio => 2,
        _ => 0,
    };

    public MiniMaxH3Transformer(MiniMaxH3Config config) => _config = config;

    public MiniMaxH3Config Config => _config;

    /// <summary>Takes ownership of the converted weights.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            _weights[kv.Key] = kv.Value;
        }
        Require("video_patch_proj.weight");
        Require("audio_patch_proj.weight");
        Require("final_layer.video_out.weight");
    }

    public IEnumerable<Tensor> EnumerateWeights() => _weights.Values;

    /// <summary>The checkpoint's own rotary base frequencies; synthesising these would shift every position.</summary>
    public float[] RopeInvFreq()
    {
        Tensor t = Require("rope.inv_freq");
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] inv = new float[f.ElementCount];
        float* p = (float*)f.DataPointer;
        for (int i = 0; i < inv.Length; i++) inv[i] = p[i];
        return inv;
    }

    private Tensor Require(string key) =>
        _weights.TryGetValue(key, out Tensor? t) ? t : throw new KeyNotFoundException($"MiniMax-H3 weight '{key}' missing.");

    private Tensor? Optional(string key) => _weights.TryGetValue(key, out Tensor? t) ? t : null;

    /// <summary>Runs the packed sequence through the token refiner, all blocks, and the dual output heads.
    /// <paramref name="videoRows"/>/<paramref name="audioRows"/> are the patchified latent rows in segment order;
    /// <paramref name="textStates"/> is the Qwen3-VL hidden state <c>[textLen, textDim]</c>.</summary>
    public (Tensor Video, Tensor Audio) Forward(IBackend backend, MiniMaxH3PackedLayout layout, Tensor videoRows,
        Tensor audioRows, Tensor textStates, Tensor cos, Tensor sin, float[] uniqueTimesteps,
        IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(layout);
        int hidden = _config.HiddenSize;
        int seq = layout.SequenceLength;

        Tensor h = new Tensor(new TensorShape(seq, hidden), DType.F32);
        try
        {
            EmbedSegments(backend, layout, h, videoRows, audioRows, textStates);
            Tensor tEmb = BuildTimeEmbedding(backend, uniqueTimesteps);
            try
            {
                List<(int Start, int Stop, int ModRow)> mods = BuildModulationSegments(layout, timestepRowOf, textTagRuns);
                for (int i = 0; i < _config.NumLayers; i++)
                {
                    ForwardBlock(backend, h, tEmb, mods, cos, sin, i);
                }
                return FinalLayer(backend, h, tEmb, layout, timestepRowOf);
            }
            finally
            {
                tEmb.Dispose();
            }
        }
        finally
        {
            h.Dispose();
        }
    }

    /// <summary>Assembles the packed stream: text rows go through condition_proj + the token refiner, video and audio
    /// rows through their patch projections, each written into its segment's slice.</summary>
    private void EmbedSegments(IBackend backend, MiniMaxH3PackedLayout layout, Tensor h, Tensor videoRows,
        Tensor audioRows, Tensor textStates)
    {
        int hidden = _config.HiddenSize;
        Tensor videoEmbed = Project(backend, videoRows, "video_patch_proj", hidden);
        Tensor audioEmbed = Project(backend, audioRows, "audio_patch_proj", hidden);
        Tensor textEmbed = RefineText(backend, textStates);
        try
        {
            int videoOffset = 0, audioOffset = 0, textOffset = 0;
            foreach (MiniMaxH3Segment seg in layout.Segments)
            {
                (Tensor src, int off) = seg.Kind switch
                {
                    MiniMaxH3SegmentKind.Text => (textEmbed, textOffset),
                    MiniMaxH3SegmentKind.Audio or MiniMaxH3SegmentKind.RefAudio => (audioEmbed, audioOffset),
                    _ => (videoEmbed, videoOffset),
                };
                CopyRows(h, seg.Start, src, off, seg.Length, hidden);
                switch (seg.Kind)
                {
                    case MiniMaxH3SegmentKind.Text: textOffset += seg.Length; break;
                    case MiniMaxH3SegmentKind.Audio or MiniMaxH3SegmentKind.RefAudio: audioOffset += seg.Length; break;
                    default: videoOffset += seg.Length; break;
                }
            }
        }
        finally
        {
            videoEmbed.Dispose();
            audioEmbed.Dispose();
            textEmbed.Dispose();
        }
    }

    private Tensor Project(IBackend backend, Tensor rows, string prefix, int outDim)
    {
        Tensor outT = new Tensor(new TensorShape(rows.Shape[0], outDim), DType.F32);
        backend.Linear(outT, rows, Require($"{prefix}.weight"), Optional($"{prefix}.bias"));
        return outT;
    }

    /// <summary>condition_proj to hidden width, then the token refiner's self-attention blocks (no modulation, no rope).</summary>
    private Tensor RefineText(IBackend backend, Tensor textStates)
    {
        int hidden = _config.HiddenSize;
        if (textStates.Shape[textStates.Shape.Rank - 1] == hidden)
        {
            Tensor passthrough = new Tensor(textStates.Shape, DType.F32);
            Buffer.MemoryCopy((void*)textStates.DataPointer, (void*)passthrough.DataPointer,
                passthrough.ElementCount * sizeof(float), textStates.ElementCount * sizeof(float));
            return passthrough;
        }
        Tensor x = Project(backend, textStates, "condition_proj", hidden);
        for (int i = 0; i < _config.TokenRefinerNumLayers; i++)
        {
            string p = $"token_refiner.blocks.{i}";
            Tensor normed = Norm(backend, x, $"{p}.norm1.weight", _config.NormEps);
            Tensor attn = Attention(backend, normed, $"{p}.attn", cos: null, sin: null);
            normed.Dispose();
            AddInPlace(x, attn);
            attn.Dispose();

            Tensor normed2 = Norm(backend, x, $"{p}.norm2.weight", _config.NormEps);
            Tensor mlp = Mlp(backend, normed2, $"{p}.mlp");
            normed2.Dispose();
            AddInPlace(x, mlp);
            mlp.Dispose();
        }
        Tensor final = Norm(backend, x, "token_refiner.final_norm.weight", _config.FinalNormEps);
        x.Dispose();
        return final;
    }

    private void ForwardBlock(IBackend backend, Tensor h, Tensor tEmb, List<(int Start, int Stop, int ModRow)> mods,
        Tensor cos, Tensor sin, int index)
    {
        string p = $"blocks.{index}";
        Tensor[] mod = Adaln(backend, tEmb, $"{p}.adaln_proj", expand: 6, modalities: 3);
        try
        {
            Tensor normed = Norm(backend, h, $"{p}.norm1.weight", _config.NormEps);
            ModulateInPlace(backend, normed, mod[0], mod[1], mods);
            Tensor attn = Attention(backend, normed, $"{p}.attn", cos, sin);
            normed.Dispose();
            GateInPlace(backend, h, attn, mod[2], mods);
            attn.Dispose();

            Tensor normed2 = Norm(backend, h, $"{p}.norm2.weight", _config.NormEps);
            ModulateInPlace(backend, normed2, mod[3], mod[4], mods);
            Tensor mlp = Mlp(backend, normed2, $"{p}.mlp");
            normed2.Dispose();
            GateInPlace(backend, h, mlp, mod[5], mods);
            mlp.Dispose();
        }
        finally
        {
            foreach (Tensor t in mod) t.Dispose();
        }
    }

    /// <summary>qkv projection, per-head q/k RMS norm, partial split-half rope, attention, output projection.</summary>
    private Tensor Attention(IBackend backend, Tensor x, string prefix, Tensor? cos, Tensor? sin)
    {
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;

        Tensor qkv = new Tensor(new TensorShape(seq, inner * 3), DType.F32);
        backend.Linear(qkv, x, Require($"{prefix}.qkv_proj.weight"), Optional($"{prefix}.qkv_proj.bias"));
        Tensor q = SplitPart(qkv, 0, seq, inner);
        Tensor k = SplitPart(qkv, 1, seq, inner);
        Tensor v = SplitPart(qkv, 2, seq, inner);
        qkv.Dispose();

        try
        {
            NormHeads(backend, q, $"{prefix}.q_norm.weight", seq, heads, hd);
            NormHeads(backend, k, $"{prefix}.k_norm.weight", seq, heads, hd);
            if (cos is not null && sin is not null)
            {
                int rotary = MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen);
                using Tensor q4 = View(q, 1, seq, heads, hd);
                using Tensor k4 = View(k, 1, seq, heads, hd);
                backend.ApplyRopeSingle(q4, cos, sin, rotary);
                backend.ApplyRopeSingle(k4, cos, sin, rotary);
            }

            Tensor qh = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            Tensor kh = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            Tensor vh = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            Tensor attn = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            try
            {
                backend.Permute0213(qh, q, seq, heads, hd);
                backend.Permute0213(kh, k, seq, heads, hd);
                backend.Permute0213(vh, v, seq, heads, hd);
                backend.ScaledDotProductAttention(attn, qh, kh, vh, null, 1f / MathF.Sqrt(hd));

                Tensor merged = new Tensor(new TensorShape(seq, inner), DType.F32);
                backend.Permute0213(merged, attn, heads, seq, hd);
                Tensor outT = new Tensor(new TensorShape(seq, _config.HiddenSize), DType.F32);
                backend.Linear(outT, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
                merged.Dispose();
                return outT;
            }
            finally
            {
                qh.Dispose(); kh.Dispose(); vh.Dispose(); attn.Dispose();
            }
        }
        finally
        {
            q.Dispose(); k.Dispose(); v.Dispose();
        }
    }

    /// <summary>Gated MLP: fc1 emits the packed gate/up pair, SwiGLU folds it, fc2 projects back.</summary>
    private Tensor Mlp(IBackend backend, Tensor x, string prefix)
    {
        int seq = (int)x.Shape[0], ffn = _config.FfnHiddenSize;
        Tensor gateUp = new Tensor(new TensorShape(seq, ffn * 2), DType.F32);
        backend.Linear(gateUp, x, Require($"{prefix}.fc1.weight"), Optional($"{prefix}.fc1.bias"));
        Tensor act = new Tensor(new TensorShape(seq, ffn), DType.F32);
        backend.GluActivate(act, gateUp, ffn, gelu: false);
        gateUp.Dispose();
        Tensor outT = new Tensor(new TensorShape(seq, _config.HiddenSize), DType.F32);
        backend.Linear(outT, act, Require($"{prefix}.fc2.weight"), Optional($"{prefix}.fc2.bias"));
        act.Dispose();
        return outT;
    }

    /// <summary>adaln projection: optional SiLU (dropped by the curve-basis checkpoints), one linear, split into
    /// <paramref name="expand"/> chunks of <c>[rows*modalities, hidden]</c>.</summary>
    private Tensor[] Adaln(IBackend backend, Tensor tEmb, string prefix, int expand, int modalities)
    {
        int rows = (int)tEmb.Shape[0], hidden = _config.HiddenSize;
        Tensor input = tEmb;
        Tensor? silu = null;
        if (!_config.UseAdalnCurves)
        {
            silu = new Tensor(tEmb.Shape, DType.F32);
            backend.Silu(silu, tEmb);
            input = silu;
        }
        Tensor proj = new Tensor(new TensorShape(rows, (long)expand * hidden * modalities), DType.F32);
        backend.Linear(proj, input, Require($"{prefix}.linear.weight"), Optional($"{prefix}.linear.bias"));
        silu?.Dispose();

        // [rows, expand*hidden*modalities] is read as [rows*modalities, expand*hidden] then split by chunk.
        Tensor[] parts = new Tensor[expand];
        int modRows = rows * modalities;
        float* src = (float*)proj.DataPointer;
        for (int e = 0; e < expand; e++)
        {
            parts[e] = new Tensor(new TensorShape(modRows, hidden), DType.F32);
            float* dst = (float*)parts[e].DataPointer;
            for (int r = 0; r < modRows; r++)
            {
                long from = (long)r * expand * hidden + (long)e * hidden;
                Buffer.MemoryCopy(src + from, dst + (long)r * hidden, hidden * sizeof(float), hidden * sizeof(float));
            }
        }
        proj.Dispose();
        return parts;
    }

    /// <summary>Per-segment <c>h = h*(1+scale) + shift</c>; the 1+ is folded on the small modulation tensor.</summary>
    private void ModulateInPlace(IBackend backend, Tensor h, Tensor shift, Tensor scale, List<(int Start, int Stop, int ModRow)> mods)
    {
        int hidden = _config.HiddenSize;
        using Tensor onePlus = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(onePlus, scale, 1f);
        foreach ((int start, int stop, int row) in mods)
        {
            using Tensor slice = RowView(h, start, stop - start, hidden);
            using Tensor s = RowView(onePlus, row, 1, hidden);
            using Tensor b = RowView(shift, row, 1, hidden);
            backend.AffineBroadcastLastDim(slice, slice, s, b);
        }
    }

    /// <summary>Per-segment gated residual <c>h += gate * value</c>.</summary>
    private void GateInPlace(IBackend backend, Tensor h, Tensor value, Tensor gate, List<(int Start, int Stop, int ModRow)> mods)
    {
        int hidden = _config.HiddenSize;
        foreach ((int start, int stop, int row) in mods)
        {
            using Tensor hs = RowView(h, start, stop - start, hidden);
            using Tensor vs = RowView(value, start, stop - start, hidden);
            using Tensor g = RowView(gate, row, 1, hidden);
            backend.GatedResidualLastDim(hs, hs, vs, g);
        }
    }

    private (Tensor Video, Tensor Audio) FinalLayer(IBackend backend, Tensor h, Tensor tEmb,
        MiniMaxH3PackedLayout layout, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf)
    {
        Tensor[] mod = Adaln(backend, tEmb, "final_layer.adaln_proj", expand: 2, modalities: 1);
        Tensor normed = Norm(backend, h, "final_layer.norm.weight", _config.FinalNormEps);
        try
        {
            MiniMaxH3Segment videoSeg = layout.Segments.Last(s => s.Kind == MiniMaxH3SegmentKind.Video);
            MiniMaxH3Segment audioSeg = layout.Segments.Last(s => s.Kind == MiniMaxH3SegmentKind.Audio);
            Tensor video = Head(backend, normed, mod[0], mod[1], videoSeg, timestepRowOf[MiniMaxH3SegmentKind.Video], "final_layer.video_out");
            Tensor audio = Head(backend, normed, mod[0], mod[1], audioSeg, timestepRowOf[MiniMaxH3SegmentKind.Audio], "final_layer.audio_out");
            return (video, audio);
        }
        finally
        {
            normed.Dispose();
            foreach (Tensor t in mod) t.Dispose();
        }
    }

    private Tensor Head(IBackend backend, Tensor normed, Tensor shift, Tensor scale, MiniMaxH3Segment seg, int row, string prefix)
    {
        int hidden = _config.HiddenSize, n = seg.Length;
        Tensor slice = new Tensor(new TensorShape(n, hidden), DType.F32);
        backend.SliceRows(slice, normed, seg.Start);
        using (Tensor onePlus = new Tensor(new TensorShape(1, hidden), DType.F32))
        using (Tensor s = RowView(scale, row, 1, hidden))
        using (Tensor b = RowView(shift, row, 1, hidden))
        {
            backend.AddScalar(onePlus, s, 1f);
            backend.AffineBroadcastLastDim(slice, slice, onePlus, b);
        }
        Tensor weight = Require($"{prefix}.weight");
        Tensor outT = new Tensor(new TensorShape(n, weight.Shape[0]), DType.F32);
        backend.Linear(outT, slice, weight, Optional($"{prefix}.bias"));
        slice.Dispose();
        return outT;
    }

    /// <summary>One modulation row per contiguous segment: <c>timestepRow*3 + modalityTag</c>. The text span is split
    /// at tag boundaries when <paramref name="textTagRuns"/> is supplied — vision pad tokens sit inside the text span
    /// but carry the VIDEO modality, so treating text as one uniform run silently mis-modulates every
    /// image/video-reference generation.</summary>
    private static List<(int Start, int Stop, int ModRow)> BuildModulationSegments(MiniMaxH3PackedLayout layout,
        IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns)
    {
        List<(int, int, int)> mods = new List<(int, int, int)>(layout.Segments.Count);
        foreach (MiniMaxH3Segment seg in layout.Segments)
        {
            int rowBase = timestepRowOf[seg.Kind] * 3;
            if (seg.Kind == MiniMaxH3SegmentKind.Text && textTagRuns is { Count: > 0 })
            {
                foreach ((int start, int stop, int tag) in textTagRuns)
                {
                    int a = seg.Start + start, b = Math.Min(seg.Start + stop, seg.Stop);
                    if (b > a)
                    {
                        mods.Add((a, b, rowBase + tag));
                    }
                }
                continue;
            }
            mods.Add((seg.Start, seg.Stop, rowBase + ModalityTag(seg.Kind)));
        }
        return mods;
    }

    /// <summary>Sinusoidal embedding (cos before sin) then proj_in/SiLU/proj_out, or a lerp of the precomputed
    /// curve table for the pruned checkpoints. Runs over the handful of distinct timesteps, not per token.</summary>
    private Tensor BuildTimeEmbedding(IBackend backend, float[] timesteps)
    {
        int rows = timesteps.Length;
        if (_config.UseAdalnCurves)
        {
            Tensor table = Require("adaln_t_table");
            int grid = (int)table.Shape[0], dim = (int)table.Shape[1];
            Tensor outT = new Tensor(new TensorShape(rows, dim), DType.F32);
            float* tp = (float*)table.DataPointer;
            float* op = (float*)outT.DataPointer;
            for (int r = 0; r < rows; r++)
            {
                double pos = Math.Clamp(timesteps[r], 0.0, 1.0) * (grid - 1);
                int i0 = Math.Min((int)Math.Floor(pos), grid - 2);
                float frac = (float)(pos - i0);
                for (int d = 0; d < dim; d++)
                {
                    float a = tp[(long)i0 * dim + d], b = tp[((long)i0 + 1) * dim + d];
                    op[(long)r * dim + d] = a + (b - a) * frac;
                }
            }
            return outT;
        }

        int freqDim = _config.TimestepInputDim, half = freqDim / 2;
        using Tensor sinusoid = new Tensor(new TensorShape(rows, freqDim), DType.F32);
        float* sp = (float*)sinusoid.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < half; i++)
            {
                double freq = Math.Exp(-Math.Log(10000.0) * i / half);
                double arg = timesteps[r] * freq;
                sp[(long)r * freqDim + i] = (float)Math.Cos(arg);
                sp[(long)r * freqDim + half + i] = (float)Math.Sin(arg);
            }
        }
        using Tensor hiddenT = new Tensor(new TensorShape(rows, _config.TimeEmbedHiddenSize), DType.F32);
        backend.Linear(hiddenT, sinusoid, Require("time_embedder.proj_in.weight"), Optional("time_embedder.proj_in.bias"));
        using Tensor act = new Tensor(hiddenT.Shape, DType.F32);
        backend.Silu(act, hiddenT);
        Tensor outEmb = new Tensor(new TensorShape(rows, _config.TimeEmbedDim), DType.F32);
        backend.Linear(outEmb, act, Require("time_embedder.proj_out.weight"), Optional("time_embedder.proj_out.bias"));
        return outEmb;
    }

    private Tensor Norm(IBackend backend, Tensor x, string weightKey, float eps)
    {
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.RmsNorm(outT, x, Require(weightKey), eps);
        return outT;
    }

    /// <summary>RMS norm applied per head over the last dim of a <c>[seq, heads*headDim]</c> buffer.</summary>
    private void NormHeads(IBackend backend, Tensor x, string weightKey, int seq, int heads, int headDim)
    {
        using Tensor view = View(x, seq * heads, headDim);
        using Tensor tmp = new Tensor(view.Shape, DType.F32);
        backend.RmsNorm(tmp, view, Require(weightKey), _config.QkNormEps);
        Buffer.MemoryCopy((void*)tmp.DataPointer, (void*)x.DataPointer,
            x.ElementCount * sizeof(float), tmp.ElementCount * sizeof(float));
    }

    private static Tensor SplitPart(Tensor qkv, int part, int seq, int inner)
    {
        Tensor outT = new Tensor(new TensorShape(seq, inner), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int r = 0; r < seq; r++)
        {
            Buffer.MemoryCopy(src + (long)r * inner * 3 + (long)part * inner, dst + (long)r * inner,
                inner * sizeof(float), inner * sizeof(float));
        }
        return outT;
    }

    private static void CopyRows(Tensor dst, int dstRow, Tensor src, int srcRow, int count, int width)
    {
        float* d = (float*)dst.DataPointer + (long)dstRow * width;
        float* s = (float*)src.DataPointer + (long)srcRow * width;
        Buffer.MemoryCopy(s, d, (long)count * width * sizeof(float), (long)count * width * sizeof(float));
    }

    private static void AddInPlace(Tensor target, Tensor addend)
    {
        float* t = (float*)target.DataPointer;
        float* a = (float*)addend.DataPointer;
        for (long i = 0; i < target.ElementCount; i++) t[i] += a[i];
    }

    private static Tensor View(Tensor t, params long[] shape) =>
        new Tensor((void*)t.DataPointer, new TensorShape(shape), t.DType);

    private static Tensor RowView(Tensor t, int row, int count, int width) =>
        new Tensor((void*)((float*)t.DataPointer + (long)row * width), new TensorShape(count, width), t.DType);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor t in _weights.Values) t.Dispose();
        _weights.Clear();
    }
}
