using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-2.3 text connectors (<c>LTX2TextConnectors</c> + per-modality <c>LTX2ConnectorTransformer1d</c>).
/// Turns the Gemma-3-12B <b>all-49-layer</b> per-token features (48 layers + embedding, <c>3840 × 49 = 188160</c>
/// per token) into the video <c>[L, 4096]</c> and audio <c>[L, 2048]</c> text embeddings the DiT cross-attends to.
/// Runs once per prompt (not per denoise step).</summary>
///
/// <remarks>Flow (LTX-2.3 <c>per_modality_projections=True</c>): per-token RMS-norm each layer's 3840-vector →
/// flatten to 188160 → zero padded tokens → scale by <c>√(streamDim / 3840)</c> → project
/// (<c>text_embedding_projection.{video,audio}_aggregate_embed</c>, 188160→4096/2048) → connector transformer:
/// replace padded positions with the 128 learnable registers (tiled to the sequence), 1D RoPE, 8 blocks of
/// (RMSNorm → gated self-attn → +res → RMSNorm → gelu FFN → +res), final RMSNorm-no-affine. The connector overwrites
/// the attention mask to all-attend, so the DiT cross-attn needs no text mask.</remarks>
public sealed unsafe class LtxVideo2TextConnectors : IDisposable
{
    private const int CaptionChannels = 3840;
    private const int ProjInFactor = 49;            // num_layers (48) + 1 embedding layer
    private const int FeatureDim = CaptionChannels * ProjInFactor;   // 188160

    private readonly int _videoDim, _audioDim;
    private readonly double _theta;
    private readonly int _baseSeqLen;
    private readonly int _numRegisters;
    private readonly int _numBlocks;

    private Tensor? _videoProjW, _videoProjB, _audioProjW, _audioProjB;
    private readonly Connector _video, _audio;

    public LtxVideo2TextConnectors(LtxVideo2Config c, int numBlocks = 8, int numRegisters = 128,
        int baseSeqLen = 4096)
    {
        _videoDim = c.InnerDim;
        _audioDim = c.AudioInnerDim;
        _theta = c.RopeTheta;
        _baseSeqLen = baseSeqLen;
        _numRegisters = numRegisters;
        _numBlocks = numBlocks;
        // The connector 1D rope follows the checkpoint-global rope_type (split for LTX-2.3); its head layout is the
        // connector attention's own (video/audio) heads.
        _video = new Connector(_videoDim, c.NumHeads, c.HeadDim, c.NormEps, c.QkNormEps, numBlocks, numRegisters,
            LtxVideo2Rope.ForConnector1d(_videoDim, _theta, baseSeqLen, c.RopeType, c.NumHeads, c.HeadDim));
        _audio = new Connector(_audioDim, c.AudioNumHeads, c.AudioHeadDim, c.NormEps, c.QkNormEps, numBlocks, numRegisters,
            LtxVideo2Rope.ForConnector1d(_audioDim, _theta, baseSeqLen, c.RopeType, c.AudioNumHeads, c.AudioHeadDim));
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _videoProjW = w["text_embedding_projection.video_aggregate_embed.weight"];
        w.TryGetValue("text_embedding_projection.video_aggregate_embed.bias", out _videoProjB);
        _audioProjW = w["text_embedding_projection.audio_aggregate_embed.weight"];
        w.TryGetValue("text_embedding_projection.audio_aggregate_embed.bias", out _audioProjB);
        _video.LoadWeights(w, "model.diffusion_model.video_embeddings_connector");
        _audio.LoadWeights(w, "model.diffusion_model.audio_embeddings_connector");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _videoProjW, _videoProjB, _audioProjW, _audioProjB })
            if (t is not null) yield return t;
        foreach (Tensor t in _video.EnumerateWeights()) yield return t;
        foreach (Tensor t in _audio.EnumerateWeights()) yield return t;
    }

    /// <summary>Projects + connects the Gemma features. <paramref name="gemmaFeatures"/> is <c>[seq, 188160]</c>
    /// (layout channel-major: feature index = <c>channel·49 + layer</c>). <paramref name="validMask"/> is
    /// <c>[seq]</c> with 1 for real tokens and 0 for padding (seq must be a multiple of the register count).
    /// Returns the video <c>[seq, 4096]</c> and audio <c>[seq, 2048]</c> text embeddings.</summary>
    public (Tensor Video, Tensor Audio) Forward(IBackend backend, Tensor gemmaFeatures, ReadOnlySpan<float> validMask)
    {
        int seq = (int)gemmaFeatures.Shape[0];
        if (seq % _numRegisters != 0)
            throw new ArgumentException($"text seq {seq} must be a multiple of {_numRegisters} registers.");

        // 1. Per-token (per-layer) RMS-norm over the 3840 channels, then zero out padded tokens.
        Tensor normed = PerTokenRmsNormMasked(gemmaFeatures, validMask, seq);

        // 2. Per-modality scale + projection + connector.
        Tensor video = ProjectAndConnect(backend, normed, validMask, seq, _videoProjW!, _videoProjB, _videoDim, _video);
        Tensor audio = ProjectAndConnect(backend, normed, validMask, seq, _audioProjW!, _audioProjB, _audioDim, _audio);
        normed.Dispose();
        return (video, audio);
    }

    private Tensor ProjectAndConnect(IBackend backend, Tensor normed, ReadOnlySpan<float> validMask, int seq,
        Tensor projW, Tensor? projB, int dim, Connector connector)
    {
        float scale = MathF.Sqrt((float)dim / CaptionChannels);
        Tensor scaled = new(new TensorShape(seq, FeatureDim), DType.F32);
        backend.Scale(scaled, normed, scale);
        Tensor proj = new(new TensorShape(seq, dim), DType.F32);
        backend.Linear(proj, scaled, projW, projB);
        scaled.Dispose();
        Tensor outT = connector.Forward(backend, proj, validMask);
        proj.Dispose();
        return outT;
    }

    /// <summary>RMS-normalizes each token's per-layer 3840-vector (variance over the channel axis, eps 1e-6), then
    /// zeros padded tokens. Input/output layout is the flattened <c>channel·49 + layer</c> ordering.</summary>
    private static Tensor PerTokenRmsNormMasked(Tensor gemma, ReadOnlySpan<float> validMask, int seq)
    {
        const float eps = 1e-6f;
        Tensor o = new(new TensorShape(seq, FeatureDim), DType.F32);
        float* gp = (float*)gemma.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int t = 0; t < seq; t++)
        {
            float* gRow = gp + (long)t * FeatureDim;
            float* oRow = op + (long)t * FeatureDim;
            if (validMask[t] == 0f)
            {
                new Span<float>(oRow, FeatureDim).Clear();
                continue;
            }
            // For each layer l, variance over the 3840 channels at stride 49.
            for (int l = 0; l < ProjInFactor; l++)
            {
                double sumSq = 0.0;
                for (int c = 0; c < CaptionChannels; c++)
                {
                    float v = gRow[(long)c * ProjInFactor + l];
                    sumSq += (double)v * v;
                }
                float inv = 1f / MathF.Sqrt((float)(sumSq / CaptionChannels) + eps);
                for (int c = 0; c < CaptionChannels; c++)
                {
                    long idx = (long)c * ProjInFactor + l;
                    oRow[idx] = gRow[idx] * inv;
                }
            }
        }
        return o;
    }

    public void Dispose()
    {
        _video.Dispose();
        _audio.Dispose();
    }

    /// <summary>One per-modality <c>LTX2ConnectorTransformer1d</c>: learnable registers + N gated self-attn blocks.</summary>
    private sealed class Connector : IDisposable
    {
        private readonly int _dim, _heads, _headDim, _numBlocks, _numRegisters;
        private readonly float _normEps;
        private readonly Tensor _ones;
        private readonly LtxVideo2Rope _rope;
        private readonly LtxVideo2Attention[] _attn;
        private Tensor? _registers;     // [numRegisters, dim]
        private Tensor?[] _ffW0, _ffB0, _ffW2, _ffB2;

        public Connector(int dim, int heads, int headDim, float normEps, float qkEps, int numBlocks, int numRegisters,
            LtxVideo2Rope rope)
        {
            _dim = dim; _heads = heads; _headDim = headDim; _normEps = normEps;
            _numBlocks = numBlocks; _numRegisters = numRegisters; _rope = rope;
            _ones = Ones(dim);
            _attn = new LtxVideo2Attention[numBlocks];
            for (int i = 0; i < numBlocks; i++) _attn[i] = new LtxVideo2Attention(dim, dim, heads, headDim, dim, qkEps);
            _ffW0 = new Tensor?[numBlocks]; _ffB0 = new Tensor?[numBlocks];
            _ffW2 = new Tensor?[numBlocks]; _ffB2 = new Tensor?[numBlocks];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _registers = LoadF32(w, $"{p}.learnable_registers");
            for (int i = 0; i < _numBlocks; i++)
            {
                _attn[i].LoadWeights(w, $"{p}.transformer_1d_blocks.{i}.attn1");
                _ffW0[i] = w[$"{p}.transformer_1d_blocks.{i}.ff.net.0.proj.weight"];
                w.TryGetValue($"{p}.transformer_1d_blocks.{i}.ff.net.0.proj.bias", out _ffB0[i]);
                _ffW2[i] = w[$"{p}.transformer_1d_blocks.{i}.ff.net.2.weight"];
                w.TryGetValue($"{p}.transformer_1d_blocks.{i}.ff.net.2.bias", out _ffB2[i]);
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_registers is not null) yield return _registers;
            for (int i = 0; i < _numBlocks; i++)
            {
                foreach (Tensor t in _attn[i].EnumerateWeights()) yield return t;
                foreach (Tensor? t in new[] { _ffW0[i], _ffB0[i], _ffW2[i], _ffB2[i] }) if (t is not null) yield return t;
            }
        }

        public Tensor Forward(IBackend backend, Tensor features, ReadOnlySpan<float> validMask)
        {
            int seq = (int)features.Shape[0];
            Tensor hidden = ReplacePaddingWithRegisters(features, validMask, seq);
            (Tensor cos, Tensor sin) = _rope.BuildConnector(seq);

            for (int i = 0; i < _numBlocks; i++)
            {
                Tensor n1 = RmsNoAffine(backend, hidden, seq);
                Tensor attn = _attn[i].Forward(backend, n1, n1, _rope, cos, sin, _rope, cos, sin, null);
                n1.Dispose();
                AddInPlace(hidden, attn); attn.Dispose();

                Tensor n2 = RmsNoAffine(backend, hidden, seq);
                Tensor ff = Ffn(backend, n2, i);
                n2.Dispose();
                AddInPlace(hidden, ff); ff.Dispose();
            }

            cos.Dispose(); sin.Dispose();
            // norm_out (RMSNorm, no affine).
            Tensor outT = RmsNoAffine(backend, hidden, seq);
            hidden.Dispose();
            return outT;
        }

        /// <summary>Gathers valid token rows to the front (order preserved) and fills the remaining positions with the
        /// tiled learnable registers (register at position <c>i</c> = <c>learnable_registers[i % numRegisters]</c>).
        /// This is the net result of the reference's gather + flipped-mask replacement, and is correct regardless of
        /// which side the text was padded on (the reference assumes Gemma's left padding; our pipeline right-pads).</summary>
        private Tensor ReplacePaddingWithRegisters(Tensor features, ReadOnlySpan<float> validMask, int seq)
        {
            float* fp = (float*)features.DataPointer;
            float* rp = (float*)_registers!.DataPointer;
            Tensor o = new(new TensorShape(seq, _dim), DType.F32);
            float* op = (float*)o.DataPointer;

            int nv = 0;
            for (int i = 0; i < seq; i++)
                if (validMask[i] != 0f)
                {
                    Buffer.MemoryCopy(fp + (long)i * _dim, op + (long)nv * _dim, (long)_dim * 4, (long)_dim * 4);
                    nv++;
                }
            for (int i = nv; i < seq; i++)
            {
                int reg = i % _numRegisters;
                Buffer.MemoryCopy(rp + (long)reg * _dim, op + (long)i * _dim, (long)_dim * 4, (long)_dim * 4);
            }
            return o;
        }

        private Tensor RmsNoAffine(IBackend backend, Tensor x, int seq)
        {
            Tensor o = new(new TensorShape(seq, _dim), DType.F32);
            backend.RmsNorm(o, x, _ones, _normEps);
            return o;
        }

        private Tensor Ffn(IBackend backend, Tensor x, int i)
        {
            int seq = (int)x.Shape[0];
            int inner = (int)_ffW0[i]!.Shape[0];
            Tensor proj = new(new TensorShape(seq, inner), DType.F32);
            backend.Linear(proj, x, _ffW0[i]!, _ffB0[i]);
            Tensor act = new(proj.Shape, DType.F32);
            backend.Gelu(act, proj);
            proj.Dispose();
            Tensor o = new(new TensorShape(seq, _dim), DType.F32);
            backend.Linear(o, act, _ffW2[i]!, _ffB2[i]);
            act.Dispose();
            return o;
        }

        private void AddInPlace(Tensor acc, Tensor add)
        {
            float* ap = (float*)acc.DataPointer, dp = (float*)add.DataPointer;
            long n = acc.Shape.ElementCount;
            for (long e = 0; e < n; e++) ap[e] += dp[e];
        }

        private static Tensor Ones(int n)
        {
            Tensor t = new(new TensorShape(n), DType.F32);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < n; i++) p[i] = 1f;
            return t;
        }

        public void Dispose() => _ones.Dispose();
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
