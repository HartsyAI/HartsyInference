using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

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

    /// <summary>Residual-stream dtype: BF16 on a backend whose adaLN/norm ops take 16-bit activations (the reference
    /// runs the block body at the checkpoint's bf16), F32 elsewhere — CpuBackend is F32-only. The patch projections,
    /// the time embedder / adaLN curve table and the output heads stay F32 either way; those are the reference's own
    /// F32 islands and each boundary casts explicitly.</summary>
    public DType BodyDType { get; init; } = DType.F32;

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

    /// <summary>Everything except the per-block weights: patch/condition projections, token refiner (its
    /// <c>token_refiner.blocks.*</c> keys don't collide — the block prefix match is anchored to the string start),
    /// time embedder / adaLN curve table, final layer, rope table. All consumed on backend A of a sharded forward.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        foreach (KeyValuePair<string, Tensor> kv in _weights)
        {
            if (!kv.Key.StartsWith("blocks.", StringComparison.Ordinal))
            {
                yield return kv.Value;
            }
        }
    }

    /// <summary>Weights of blocks <c>[startBlock, endBlock)</c> only — the asymmetric-preload primitive for DiT
    /// sharding: backend A preloads <see cref="EnumerateSharedWeights"/> + its range, backend B ONLY its range.</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        foreach (KeyValuePair<string, Tensor> kv in _weights)
        {
            if (TryParseBlockIndex(kv.Key, out int block) && block >= startBlock && block < endBlock)
            {
                yield return kv.Value;
            }
        }
    }

    private static bool TryParseBlockIndex(string key, out int block)
    {
        block = -1;
        if (!key.StartsWith("blocks.", StringComparison.Ordinal))
        {
            return false;
        }
        int dot = key.IndexOf('.', 7);
        return dot > 7 && int.TryParse(key.AsSpan(7, dot - 7), out block);
    }

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
        int seq = layout.SequenceLength;
        // Resolved once per forward, not per block: free VRAM only needs a cheap driver query (no stream sync, so
        // this cannot affect the GPU-resident-loop D2H-sync count), but re-querying inside the block loop would let
        // the chunk size drift mid-generation as VRAM usage elsewhere changes — non-deterministic and hostile to
        // future CUDA-graph capture of this loop.
        (long freeBytes, _) = backend.GetVramInfo();
        int chunkRows = MiniMaxH3ChunkPolicy.ResolveChunkRows(seq, _config, BodyDType, freeBytes);
        if (chunkRows != int.MaxValue)
        {
            Logs.Info($"[MiniMaxH3] chunked attention/MLP: seq={seq} chunkRows={chunkRows} freeBytes={freeBytes}.");
        }

        Tensor h = EmbedSegments(backend, layout, videoRows, audioRows, textStates);
        try
        {
            Tensor tEmb = BuildTimeEmbedding(backend, uniqueTimesteps);
            Tensor modIndex = BuildModulationIndex(layout, seq, timestepRowOf, textTagRuns);
            try
            {
                for (int i = 0; i < _config.NumLayers; i++)
                {
                    ForwardBlock(backend, ref h, tEmb, modIndex, cos, sin, i, chunkRows);
                }
                return FinalLayer(backend, h, tEmb, layout, timestepRowOf);
            }
            finally
            {
                modIndex.Dispose();
                tEmb.Dispose();
            }
        }
        finally
        {
            h.Dispose();
        }
    }

    /// <summary>DiT-sharded forward: blocks <c>[0, splitBlock)</c> on <paramref name="backendA"/> (which also owns
    /// every shared weight — embeds, token refiner, final layer), <c>[splitBlock, NumLayers)</c> on
    /// <paramref name="backendB"/>. The residual stream <c>h</c> and the tiny time embedding cross via
    /// <see cref="IBackend.CopyFromPeer"/>; the host-built modulation index and rope cos/sin tables are consumed
    /// by both backends directly (each stages its own upload — sequential use, never concurrent). VRAM pooling,
    /// not latency: this is what makes the ~19.5 GB fp8 build fully resident across a 24+12 GB pair instead of
    /// streaming ~19 GB of weights per step. H3 has no step-graph/step-cache/block-streaming, so there are no
    /// exclusions to gate — the fp8-build-only constraint lives in the recipe's existing &lt;40 GB check.</summary>
    public (Tensor Video, Tensor Audio) ForwardSharded(IBackend backendA, IBackend backendB,
        MiniMaxH3PackedLayout layout, Tensor videoRows, Tensor audioRows, Tensor textStates, Tensor cos, Tensor sin,
        float[] uniqueTimesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf, int splitBlock,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns = null)
    {
        ArgumentNullException.ThrowIfNull(backendA);
        ArgumentNullException.ThrowIfNull(backendB);
        ArgumentNullException.ThrowIfNull(layout);
        if (splitBlock <= 0 || splitBlock >= _config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {_config.NumLayers}) exclusive, got {splitBlock}.");
        int seq = layout.SequenceLength;
        // Per-backend: the shard's smaller card needs a smaller chunk than the primary. See Forward's identical
        // comment on why this is resolved once per forward rather than per block.
        (long freeA, _) = backendA.GetVramInfo();
        (long freeB, _) = backendB.GetVramInfo();
        int chunkRowsA = MiniMaxH3ChunkPolicy.ResolveChunkRows(seq, _config, BodyDType, freeA);
        int chunkRowsB = MiniMaxH3ChunkPolicy.ResolveChunkRows(seq, _config, BodyDType, freeB);
        if (chunkRowsA != int.MaxValue || chunkRowsB != int.MaxValue)
        {
            Logs.Info($"[MiniMaxH3] chunked attention/MLP (sharded): seq={seq} chunkRowsA={chunkRowsA} "
                + $"freeBytesA={freeA} chunkRowsB={chunkRowsB} freeBytesB={freeB}.");
        }

        Tensor h = EmbedSegments(backendA, layout, videoRows, audioRows, textStates);
        Tensor tEmb = BuildTimeEmbedding(backendA, uniqueTimesteps);
        Tensor modIndex = BuildModulationIndex(layout, seq, timestepRowOf, textTagRuns);
        try
        {
            for (int i = 0; i < splitBlock; i++)
            {
                ForwardBlock(backendA, ref h, tEmb, modIndex, cos, sin, i, chunkRowsA);
            }

            // Boundary A→B: the residual stream MOVES; tEmb is only COPIED (the final layer on A reads it after
            // B's range). The curve-table tEmb variant is host-built, but the sinusoidal variant is device-built
            // on A — copy unconditionally so both checkpoint families behave identically.
            Tensor hB = CopyAcross(backendB, backendA, h);
            h.Dispose();
            h = hB;
            Tensor tEmbB = CopyAcross(backendB, backendA, tEmb);
            try
            {
                for (int i = splitBlock; i < _config.NumLayers; i++)
                {
                    ForwardBlock(backendB, ref h, tEmbB, modIndex, cos, sin, i, chunkRowsB);
                }
            }
            finally
            {
                tEmbB.Dispose();
            }

            // Boundary B→A: the final layer's weights live in the shared set on A.
            Tensor hBack = CopyAcross(backendA, backendB, h);
            h.Dispose();
            h = hBack;
            return FinalLayer(backendA, h, tEmb, layout, timestepRowOf);
        }
        finally
        {
            modIndex.Dispose();
            tEmb.Dispose();
            h.Dispose();
        }
    }

    /// <summary>Peer-copies <paramref name="source"/> onto <paramref name="dst"/>'s device; the source stays live.</summary>
    private static Tensor CopyAcross(IBackend dst, IBackend src, Tensor source)
    {
        Tensor copied = new Tensor(source.Shape, source.DType);
        dst.CopyFromPeer(copied, source, src);
        return copied;
    }

    /// <summary>Assembles the packed stream: text rows go through condition_proj + the token refiner, video and audio
    /// rows through their patch projections, concatenated in segment order. The residual stream is shaped
    /// <c>[seq, 1, hidden]</c> so the adaLN ops see a broadcast group of one row and modulate per token.</summary>
    private Tensor EmbedSegments(IBackend backend, MiniMaxH3PackedLayout layout, Tensor videoRows,
        Tensor audioRows, Tensor textStates)
    {
        int hidden = _config.HiddenSize;
        Tensor videoEmbed = ProjectIntoStream(backend, videoRows, "video_patch_proj", hidden);
        Tensor audioEmbed = ProjectIntoStream(backend, audioRows, "audio_patch_proj", hidden);
        Tensor textEmbed = RefineText(backend, textStates);
        List<Tensor> pieces = new List<Tensor>(layout.Segments.Count);
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
                if (seg.Length > 0)
                {
                    Tensor piece = new Tensor(new TensorShape(seg.Length, hidden), BodyDType);
                    pieces.Add(piece);
                    backend.SliceRows(piece, src, off);
                }
                switch (seg.Kind)
                {
                    case MiniMaxH3SegmentKind.Text: textOffset += seg.Length; break;
                    case MiniMaxH3SegmentKind.Audio or MiniMaxH3SegmentKind.RefAudio: audioOffset += seg.Length; break;
                    default: videoOffset += seg.Length; break;
                }
            }
            Tensor h = new Tensor(new TensorShape(layout.SequenceLength, 1, hidden), BodyDType);
            backend.Concat(h, CollectionsMarshal.AsSpan(pieces), 0);
            return h;
        }
        finally
        {
            foreach (Tensor t in pieces) t.Dispose();
            videoEmbed.Dispose();
            audioEmbed.Dispose();
            textEmbed.Dispose();
        }
    }

    private Tensor Project(IBackend backend, Tensor rows, string prefix, int outDim, DType dtype)
    {
        Tensor outT = new Tensor(new TensorShape(rows.Shape[0], outDim), dtype);
        backend.Linear(outT, rows, Require($"{prefix}.weight"), Optional($"{prefix}.bias"));
        return outT;
    }

    /// <summary>Input projection into the residual stream: the GEMM runs F32 (the reference's patch projections are
    /// an F32 island) and only its result casts in — a biased Linear whose operands resolve to F32 while its output
    /// is 16-bit lands on the cuBLASLt bias epilogue's unsupported dtype pair and returns garbage.</summary>
    private Tensor ProjectIntoStream(IBackend backend, Tensor rows, string prefix, int outDim)
        => DtypeCastHelper.EnsureDtype(backend, Project(backend, rows, prefix, outDim, DType.F32), BodyDType);

    /// <summary>condition_proj to hidden width, then the token refiner's self-attention blocks (no modulation, no rope).</summary>
    private Tensor RefineText(IBackend backend, Tensor textStates)
    {
        int hidden = _config.HiddenSize;
        if (textStates.Shape[textStates.Shape.Rank - 1] == hidden)
        {
            Tensor passthrough = new Tensor(textStates.Shape, textStates.DType);
            backend.SliceRows(passthrough, textStates, 0);
            return DtypeCastHelper.EnsureDtype(backend, passthrough, BodyDType);
        }
        Tensor x = ProjectIntoStream(backend, textStates, "condition_proj", hidden);
        for (int i = 0; i < _config.TokenRefinerNumLayers; i++)
        {
            string p = $"token_refiner.blocks.{i}";
            Tensor normed = Norm(backend, x, $"{p}.norm1.weight", _config.NormEps, BodyDType);
            Tensor attn = Attention(backend, normed, $"{p}.attn", cos: null, sin: null);
            normed.Dispose();
            x = Residual(backend, x, attn);

            Tensor normed2 = Norm(backend, x, $"{p}.norm2.weight", _config.NormEps, BodyDType);
            Tensor mlp = Mlp(backend, normed2, $"{p}.mlp");
            normed2.Dispose();
            x = Residual(backend, x, mlp);
        }
        Tensor final = Norm(backend, x, "token_refiner.final_norm.weight", _config.FinalNormEps, BodyDType);
        x.Dispose();
        return final;
    }

    private void ForwardBlock(IBackend backend, ref Tensor h, Tensor tEmb, Tensor modIndex,
        Tensor cos, Tensor sin, int index, int chunkRows)
    {
        string p = $"blocks.{index}";
        int seq = (int)h.Shape[0];
        Tensor[] mod = Adaln(backend, tEmb, $"{p}.adaln_proj", expand: 6, modalities: 3);
        try
        {
            Tensor normed = Norm(backend, h, $"{p}.norm1.weight", _config.NormEps, BodyDType);
            Tensor modulated = Modulate(backend, normed, mod[0], mod[1], modIndex,
                Optional($"{p}.attn.qkv_proj.weight"));
            normed.Dispose();
            Tensor attn = seq > chunkRows
                ? AttentionChunked(backend, modulated, $"{p}.attn", cos, sin, chunkRows)
                : Attention(backend, modulated, $"{p}.attn", cos, sin);
            modulated.Dispose();
            Tensor gatedAttn = Gate(backend, h, attn, mod[2], modIndex);
            attn.Dispose();
            Swap(ref h, gatedAttn);

            Tensor normed2 = Norm(backend, h, $"{p}.norm2.weight", _config.NormEps, BodyDType);
            Tensor modulated2 = Modulate(backend, normed2, mod[3], mod[4], modIndex,
                Optional($"{p}.mlp.fc1.weight"));
            normed2.Dispose();
            Tensor mlp = seq > chunkRows
                ? MlpChunked(backend, modulated2, $"{p}.mlp", chunkRows)
                : Mlp(backend, modulated2, $"{p}.mlp");
            modulated2.Dispose();
            Tensor gatedMlp = Gate(backend, h, mlp, mod[5], modIndex);
            mlp.Dispose();
            Swap(ref h, gatedMlp);
        }
        finally
        {
            foreach (Tensor t in mod) t.Dispose();
        }
    }

    /// <summary>qkv projection, per-head q/k RMS norm, partial split-half rope, attention, output projection.
    /// q/k/v are allocated head-major <c>[1, heads, seq, headDim]</c> up front so the fused split+norm, the in-place
    /// rope and SDPA all consume them directly — only the attention output needs a permute back.
    /// The qkv buffer through to the attention output stays F32: rope and SDPA have no 16-bit path, so a bf16 qkv
    /// would only add casts around them. Only the projection's input and output ride the bf16 residual stream.</summary>
    private Tensor Attention(IBackend backend, Tensor x, string prefix, Tensor? cos, Tensor? sin)
    {
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;
        TensorShape headed = new TensorShape(1, heads, seq, hd);

        Tensor q = new Tensor(headed, DType.F32);
        Tensor k = new Tensor(headed, DType.F32);
        Tensor v = new Tensor(headed, DType.F32);
        try
        {
            using (Tensor qkv = new Tensor(new TensorShape(seq, inner * 3), DType.F32))
            {
                backend.Linear(qkv, x, Require($"{prefix}.qkv_proj.weight"), Optional($"{prefix}.qkv_proj.bias"));
                // One kernel for the 3-way split plus both per-head norms, emitting head-major so SDPA consumes
                // q/k/v directly — no permute.
                backend.QkvSplitNormHeadMajor(q, k, v, qkv, Require($"{prefix}.q_norm.weight"),
                    Require($"{prefix}.k_norm.weight"), _config.QkNormEps);
            }
            if (cos is not null && sin is not null)
            {
                int rotary = MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen);
                backend.ApplyRopeSingleHeadMajor(q, cos, sin, rotary);
                backend.ApplyRopeSingleHeadMajor(k, cos, sin, rotary);
            }

            Tensor attn = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            try
            {
                backend.ScaledDotProductAttention(attn, q, k, v, null, 1f / MathF.Sqrt(hd));

                Tensor merged = new Tensor(new TensorShape(seq, inner), DType.F32);
                backend.Permute0213(merged, attn, heads, seq, hd);
                // Body dtype, not x's: x may be the fp8 form emitted by the fused modulate, and out_proj's
                // result feeds the residual stream, not another fp8 Linear.
                Tensor outT = new Tensor(x.Shape, BodyDType);
                backend.Linear(outT, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
                merged.Dispose();
                return outT;
            }
            finally
            {
                attn.Dispose();
            }
        }
        finally
        {
            q.Dispose(); k.Dispose(); v.Dispose();
        }
    }

    /// <summary>Row-chunked equivalent of <see cref="Attention"/> for a long sequence: the packed <c>qkv</c> buffer
    /// (<c>[seq, inner*3]</c>) and the full head-major <c>q/k/v</c> would otherwise be simultaneously live at
    /// ~20 GiB combined at H3's largest requested geometries. Two passes instead: (1) project + split + RoPE one
    /// chunk of queries/keys/values at a time, concatenating <c>k</c>/<c>v</c> into full tensors (every query must
    /// see every key — this part is irreducible) while keeping each chunk's <c>q</c> in a list for pass 2; (2) run
    /// SDPA per query chunk against the full <c>k</c>/<c>v</c> (SDPA already supports Sq≠Skv — this is not a new
    /// capability, just an external call shape it was never driven at) so the full-size <c>attn</c> output is never
    /// materialized either, then permute+project+concat each chunk's output. Never called directly — see
    /// <see cref="ForwardBlock"/>'s dispatch, which sends anything under <c>chunkRows</c> to the exact original
    /// <see cref="Attention"/> path so small/legacy-path calls stay bit-identical (chunked GEMMs pick different
    /// cuBLASLt algorithms and are not bitwise equal to a single full-width GEMM).</summary>
    private Tensor AttentionChunked(IBackend backend, Tensor x, string prefix, Tensor? cos, Tensor? sin, int chunkRows)
    {
        int seq = (int)x.Shape[0];
        int heads = _config.NumAttentionHeads, hd = _config.AttentionHeadDim, inner = heads * hd;
        Tensor qkvW = Require($"{prefix}.qkv_proj.weight");
        Tensor qNormW = Require($"{prefix}.q_norm.weight");
        Tensor kNormW = Require($"{prefix}.k_norm.weight");
        Tensor? qkvB = Optional($"{prefix}.qkv_proj.bias");
        int? rotary = cos is not null && sin is not null ? MiniMaxH3Rope.RotaryDim(_config.RopeInvFreqLen) : null;

        List<Tensor> qChunks = new List<Tensor>();
        List<Tensor> kChunks = new List<Tensor>();
        List<Tensor> vChunks = new List<Tensor>();
        try
        {
            for (int r0 = 0; r0 < seq; r0 += chunkRows)
            {
                int c = Math.Min(chunkRows, seq - r0);
                using Tensor xChunk = new Tensor(WithFirstDim(x.Shape, c), x.DType) { Fp8ScaleFactor = x.Fp8ScaleFactor };
                backend.SliceRowsGeneric(xChunk, x, r0);
                Tensor qc = new Tensor(new TensorShape(1, heads, c, hd), DType.F32);
                Tensor kc = new Tensor(new TensorShape(1, heads, c, hd), DType.F32);
                Tensor vc = new Tensor(new TensorShape(1, heads, c, hd), DType.F32);
                using (Tensor qkvChunk = new Tensor(new TensorShape(c, inner * 3), DType.F32))
                {
                    backend.Linear(qkvChunk, xChunk, qkvW, qkvB);
                    backend.QkvSplitNormHeadMajor(qc, kc, vc, qkvChunk, qNormW, kNormW, _config.QkNormEps);
                }
                if (rotary is int rot)
                {
                    using Tensor cosChunk = new Tensor(new TensorShape(c, cos!.Shape[cos.Shape.Rank - 1]), DType.F32);
                    using Tensor sinChunk = new Tensor(new TensorShape(c, sin!.Shape[sin.Shape.Rank - 1]), DType.F32);
                    backend.SliceRows(cosChunk, cos, r0);
                    backend.SliceRows(sinChunk, sin, r0);
                    backend.ApplyRopeSingleHeadMajor(qc, cosChunk, sinChunk, rot);
                    backend.ApplyRopeSingleHeadMajor(kc, cosChunk, sinChunk, rot);
                }
                qChunks.Add(qc);
                kChunks.Add(kc);
                vChunks.Add(vc);
            }

            Tensor kFull = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            Tensor vFull = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            backend.Concat(kFull, CollectionsMarshal.AsSpan(kChunks), 2);
            backend.Concat(vFull, CollectionsMarshal.AsSpan(vChunks), 2);
            foreach (Tensor t in kChunks) t.Dispose();
            foreach (Tensor t in vChunks) t.Dispose();
            kChunks.Clear();
            vChunks.Clear();

            try
            {
                List<Tensor> outChunks = new List<Tensor>(qChunks.Count);
                try
                {
                    for (int i = 0; i < qChunks.Count; i++)
                    {
                        int c = (int)qChunks[i].Shape[2];
                        using Tensor attnChunk = new Tensor(new TensorShape(1, heads, c, hd), DType.F32);
                        backend.ScaledDotProductAttention(attnChunk, qChunks[i], kFull, vFull, null, 1f / MathF.Sqrt(hd));
                        qChunks[i].Dispose();
                        using Tensor merged = new Tensor(new TensorShape(c, inner), DType.F32);
                        backend.Permute0213(merged, attnChunk, heads, c, hd);
                        Tensor outChunk = new Tensor(WithFirstDim(x.Shape, c), BodyDType);
                        backend.Linear(outChunk, merged, Require($"{prefix}.out_proj.weight"), Optional($"{prefix}.out_proj.bias"));
                        outChunks.Add(outChunk);
                    }
                    qChunks.Clear();
                    Tensor outT = new Tensor(x.Shape, BodyDType);
                    backend.Concat(outT, CollectionsMarshal.AsSpan(outChunks), 0);
                    return outT;
                }
                finally
                {
                    foreach (Tensor t in outChunks) t.Dispose();
                }
            }
            finally
            {
                kFull.Dispose();
                vFull.Dispose();
            }
        }
        finally
        {
            foreach (Tensor t in qChunks) t.Dispose();
            foreach (Tensor t in kChunks) t.Dispose();
            foreach (Tensor t in vChunks) t.Dispose();
        }
    }

    /// <summary>Gated MLP: fc1 emits the packed gate/up pair, SwiGLU folds it, fc2 projects back. The gate/up pair
    /// stays F32 — GluActivate has an F32 and a BF16 kernel but no F16 one — so only fc2's result rejoins the
    /// stream at the body dtype.</summary>
    private Tensor Mlp(IBackend backend, Tensor x, string prefix)
    {
        int seq = (int)x.Shape[0], ffn = _config.FfnHiddenSize;
        Tensor gateUp = new Tensor(new TensorShape(seq, ffn * 2), DType.F32);
        backend.Linear(gateUp, x, Require($"{prefix}.fc1.weight"), Optional($"{prefix}.fc1.bias"));
        Tensor act = new Tensor(new TensorShape(seq, ffn), DType.F32);
        backend.GluActivate(act, gateUp, ffn, gelu: false);
        gateUp.Dispose();
        Tensor outT = new Tensor(x.Shape, BodyDType);
        backend.Linear(outT, act, Require($"{prefix}.fc2.weight"), Optional($"{prefix}.fc2.bias"));
        act.Dispose();
        return outT;
    }

    /// <summary>Row-chunked equivalent of <see cref="Mlp"/>: <c>gateUp [seq, ffn*2]</c> and <c>act [seq, ffn]</c>
    /// have no cross-token dependency at all (a plain per-row FFN), so chunking is a direct per-chunk repeat of the
    /// same three ops with no reassembly step beyond the final <see cref="IBackend.Concat"/> — unlike attention,
    /// there is no full-size intermediate this needs to keep resident. Never called directly — see
    /// <see cref="ForwardBlock"/>'s <c>chunkRows</c> dispatch and <see cref="AttentionChunked"/>'s doc for why the
    /// unchunked <see cref="Mlp"/> path must stay untouched for bit-exactness on small/legacy-path calls.</summary>
    private Tensor MlpChunked(IBackend backend, Tensor x, string prefix, int chunkRows)
    {
        int seq = (int)x.Shape[0], ffn = _config.FfnHiddenSize;
        Tensor fc1W = Require($"{prefix}.fc1.weight");
        Tensor? fc1B = Optional($"{prefix}.fc1.bias");
        Tensor fc2W = Require($"{prefix}.fc2.weight");
        Tensor? fc2B = Optional($"{prefix}.fc2.bias");

        List<Tensor> outChunks = new List<Tensor>();
        try
        {
            for (int r0 = 0; r0 < seq; r0 += chunkRows)
            {
                int c = Math.Min(chunkRows, seq - r0);
                using Tensor xChunk = new Tensor(WithFirstDim(x.Shape, c), x.DType) { Fp8ScaleFactor = x.Fp8ScaleFactor };
                backend.SliceRowsGeneric(xChunk, x, r0);
                using Tensor gateUpChunk = new Tensor(new TensorShape(c, ffn * 2), DType.F32);
                backend.Linear(gateUpChunk, xChunk, fc1W, fc1B);
                using Tensor actChunk = new Tensor(new TensorShape(c, ffn), DType.F32);
                backend.GluActivate(actChunk, gateUpChunk, ffn, gelu: false);
                Tensor outChunk = new Tensor(WithFirstDim(x.Shape, c), BodyDType);
                backend.Linear(outChunk, actChunk, fc2W, fc2B);
                outChunks.Add(outChunk);
            }
            Tensor outT = new Tensor(x.Shape, BodyDType);
            backend.Concat(outT, CollectionsMarshal.AsSpan(outChunks), 0);
            return outT;
        }
        finally
        {
            foreach (Tensor t in outChunks) t.Dispose();
        }
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
        // Linear sizes itself from the weight and the input row count, so shaping its output
        // [rows*modalities, expand*hidden] is the free reinterpret the chunk split needs.
        int modRows = rows * modalities;
        using Tensor proj = new Tensor(new TensorShape(modRows, (long)expand * hidden), DType.F32);
        backend.Linear(proj, input, Require($"{prefix}.linear.weight"), Optional($"{prefix}.linear.bias"));
        silu?.Dispose();

        Tensor[] parts = new Tensor[expand];
        for (int e = 0; e < expand; e++)
        {
            parts[e] = new Tensor(new TensorShape(modRows, hidden), DType.F32);
            backend.SliceLastDim(parts[e], proj, e * hidden);
        }
        return parts;
    }

    /// <summary>Per-token <c>out = h*(1+scale) + shift</c>. The modulation table is indexed per token inside the
    /// kernel, so nothing is materialized at <c>[seq, hidden]</c> and the table stays F32 while the stream is bf16.</summary>
    private static Tensor Modulate(IBackend backend, Tensor h, Tensor shift, Tensor scale, Tensor modIndex,
        Tensor? consumerWeight = null)
    {
        // The modulated tensor's only reader is the fp8 Linear right after it, so when that Linear carries a
        // static activation scale the producer can write e4m3 directly and the F32 form never exists. Saves a
        // full-width write plus the quantize kernel's full-width read at each site.
        if (consumerWeight is not null && consumerWeight.DType.IsFp8 && consumerWeight.Fp8InputScaleFactor > 0f
            && h.DType == DType.F32)
        {
            Tensor fp8 = new Tensor(h.Shape, DType.F8E4M3) { Fp8ScaleFactor = consumerWeight.Fp8InputScaleFactor };
            if (backend.TryAffineBroadcastRowIndexedToFp8(fp8, h, scale, shift, modIndex, consumerWeight))
            {
                return fp8;
            }
            fp8.Dispose();
        }
        Tensor outT = new Tensor(h.Shape, h.DType);
        backend.AffineBroadcastRowIndexed(outT, h, scale, shift, modIndex);
        return outT;
    }

    /// <summary>Per-token gated residual <c>out = h + gate * value</c>; see <see cref="Modulate"/> for the layout.</summary>
    private static Tensor Gate(IBackend backend, Tensor h, Tensor value, Tensor gate, Tensor modIndex)
    {
        Tensor outT = new Tensor(h.Shape, h.DType);
        backend.GatedResidualRowIndexed(outT, h, value, gate, modIndex);
        return outT;
    }

    /// <summary>Copies <paramref name="shape"/> with its first dimension replaced by <paramref name="d0"/> — the
    /// chunk loops' shape builder, so a chunk tensor matches its parent's rank exactly (the residual stream carries
    /// a harmless singleton middle dim, <c>[seq, 1, hidden]</c>, that every op already treats as part of the row).</summary>
    private static TensorShape WithFirstDim(TensorShape shape, long d0)
    {
        Span<long> dims = stackalloc long[shape.Rank];
        for (int i = 0; i < shape.Rank; i++) dims[i] = shape[i];
        dims[0] = d0;
        return new TensorShape(dims);
    }

    /// <summary>Plain residual add into a fresh tensor (device backends bind an op's result to the tensor it was handed).</summary>
    private static Tensor Residual(IBackend backend, Tensor x, Tensor addend)
    {
        Tensor outT = new Tensor(x.Shape, x.DType);
        backend.Add(outT, x, addend);
        x.Dispose();
        addend.Dispose();
        return outT;
    }

    private (Tensor Video, Tensor Audio) FinalLayer(IBackend backend, Tensor h, Tensor tEmb,
        MiniMaxH3PackedLayout layout, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf)
    {
        Tensor[] mod = Adaln(backend, tEmb, "final_layer.adaln_proj", expand: 2, modalities: 1);
        // The heads are the checkpoint's F32 island, so the stream leaves bf16 here — one cast for the whole tail.
        Tensor normed = Norm(backend, h, "final_layer.norm.weight", _config.FinalNormEps, DType.F32);
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
        Tensor modulated = new Tensor(new TensorShape(n, hidden), DType.F32);
        using (Tensor slice = new Tensor(new TensorShape(n, hidden), DType.F32))
        using (Tensor onePlus = new Tensor(new TensorShape(1, hidden), DType.F32))
        using (Tensor s = new Tensor(new TensorShape(1, hidden), DType.F32))
        using (Tensor b = new Tensor(new TensorShape(1, hidden), DType.F32))
        {
            backend.SliceRows(slice, normed, seg.Start);
            backend.SliceRows(s, scale, row);
            backend.SliceRows(b, shift, row);
            backend.AddScalar(onePlus, s, 1f);
            backend.AffineBroadcastLastDim(modulated, slice, onePlus, b);
        }
        Tensor weight = Require($"{prefix}.weight");
        Tensor outT = new Tensor(new TensorShape(n, weight.Shape[0]), DType.F32);
        backend.Linear(outT, modulated, weight, Optional($"{prefix}.bias"));
        modulated.Dispose();
        return outT;
    }

    /// <summary>One modulation row per token — <c>timestepRow*3 + modalityTag</c> — as the I32 gather index the
    /// adaLN ops expand through. The text span is split at tag boundaries when <paramref name="textTagRuns"/> is
    /// supplied: vision pad tokens sit inside the text span but carry the VIDEO modality, so treating text as one
    /// uniform run silently mis-modulates every image/video-reference generation.</summary>
    private static Tensor BuildModulationIndex(MiniMaxH3PackedLayout layout, int seq,
        IReadOnlyDictionary<MiniMaxH3SegmentKind, int> timestepRowOf,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns)
    {
        Tensor index = new Tensor(new TensorShape(seq), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int i = 0; i < seq; i++) p[i] = -1;
        foreach (MiniMaxH3Segment seg in layout.Segments)
        {
            int rowBase = timestepRowOf[seg.Kind] * 3;
            if (seg.Kind == MiniMaxH3SegmentKind.Text && textTagRuns is { Count: > 0 })
            {
                foreach ((int start, int stop, int tag) in textTagRuns)
                {
                    int a = seg.Start + start, b = Math.Min(seg.Start + stop, seg.Stop);
                    for (int i = a; i < b; i++) p[i] = rowBase + tag;
                }
                continue;
            }
            for (int i = seg.Start; i < seg.Stop; i++) p[i] = rowBase + ModalityTag(seg.Kind);
        }
        for (int i = 0; i < seq; i++)
        {
            if (p[i] < 0)
            {
                index.Dispose();
                throw new HartsyInferenceException(
                    $"MiniMax-H3 packed row {i} of {seq} has no modulation segment; the tag runs do not cover the text span.");
            }
        }
        return index;
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

    private Tensor Norm(IBackend backend, Tensor x, string weightKey, float eps, DType dtype)
    {
        Tensor outT = new Tensor(x.Shape, dtype);
        backend.RmsNorm(outT, x, Require(weightKey), eps);
        return outT;
    }

    /// <summary>Replaces the residual stream, disposing the tensor it displaces.</summary>
    private static void Swap(ref Tensor current, Tensor replacement)
    {
        Tensor old = current;
        current = replacement;
        old.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor t in _weights.Values) t.Dispose();
        _weights.Clear();
    }
}
