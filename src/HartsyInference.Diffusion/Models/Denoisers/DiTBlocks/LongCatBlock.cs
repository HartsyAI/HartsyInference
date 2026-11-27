using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LongCat-Video DiT block (<c>LongCatSingleStreamBlock</c>, meituan-longcat/LongCat-Video @ 58eb336). B=1 over <c>[S, dim]</c> with a
/// per-frame modulation: (1) self-attention with fused QKV, per-head QK RMSNorm and 3D RoPE, run as the segments the caller plans, gated;
/// (2) ungated text cross-attention on the noisy rows only, after an affine LayerNorm; (3) SwiGLU FFN, gated. Every weight is used through
/// <see cref="IBackend.Linear"/> or <see cref="IBackend.LinearWeightRows"/>, which apply a runtime LoRA adjunct on any projection, and no
/// path casts or fuses a weight past that.</summary>
public sealed partial class LongCatBlock : IStreamingBlock
{
    private readonly LongCatConfig _config;
    private Tensor? _adaWeight, _adaBias;
    // The reference's x is BF16 on a BF16 or INT8 release, so each residual sum rounds to BF16 there; the F32 route (tiny F32 weights) does not.
    private bool _bf16Residual;
    private Tensor? _qkv, _qkvBias, _qNorm, _kNorm, _proj, _projBias;
    private Tensor? _crossNormWeight, _crossNormBias;
    private Tensor? _crossQ, _crossQBias, _crossKv, _crossKvBias, _crossQNorm, _crossKNorm, _crossProj, _crossProjBias;
    private Tensor? _w1, _w2, _w3;
    private AttentionLayer? _selfAttention, _crossAttention;

    /// <summary>Rows per tile for the FFN (the SwiGLU intermediates, [rows, FFN] F32 each) and for the self-attention QKV projection ([rows, 3·dim]).
    /// Untiled at 37,440 tokens these are 1.57 GB and 1.84 GB apiece; tiling bounds them to one tile. One constant for both, so the headroom formula
    /// reads the same value the blocks run with.</summary>
    public const int DefaultTileRows = 4096;

    private int _ffnTileRows = DefaultTileRows;

    /// <summary>Sets the FFN tile height. A value at or above the token count runs the FFN untiled; tests use small tiles to exercise the seams.</summary>
    internal void SetFfnTileRows(int rows)
    {
        if (rows < 1) throw new ArgumentOutOfRangeException(nameof(rows));
        _ffnTileRows = rows;
    }
    private LongCatAvatarAudioStep? _audio;

    /// <summary>Creates a block of the given geometry; <see cref="LoadWeights"/> fills it.</summary>
    public LongCatBlock(LongCatConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
    }

    /// <inheritdoc/>
    public long EstimatedWeightBytes { get; private set; }

    /// <summary>Reads the block's tensors under <paramref name="prefix"/> (<c>blocks.N</c>).</summary>
    /// <param name="int8Release">True when the checkpoint is the per-row INT8 release. Its linears are widened before this call, so the release is
    /// read from the checkpoint, not from the widened tensors' dtype.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned, bool int8Release = false)
    {
        _adaWeight = w[$"{prefix}.adaLN_modulation.1.weight"];
        _adaBias = LongCatOps.WidenF32(w, $"{prefix}.adaLN_modulation.1.bias", owned);
        _qkv = w[$"{prefix}.attn.qkv.weight"];
        _bf16Residual = EngineKnobs.LongcatBf16Residual.Value && (int8Release || _qkv.DType == DType.BF16);
        _qkvBias = LongCatOps.WidenF32(w, $"{prefix}.attn.qkv.bias", owned);
        _qNorm = LongCatOps.WidenF32(w, $"{prefix}.attn.q_norm.weight", owned);
        _kNorm = LongCatOps.WidenF32(w, $"{prefix}.attn.k_norm.weight", owned);
        _proj = w[$"{prefix}.attn.proj.weight"];
        _projBias = LongCatOps.WidenF32(w, $"{prefix}.attn.proj.bias", owned);
        _crossNormWeight = LongCatOps.WidenF32(w, $"{prefix}.pre_crs_attn_norm.weight", owned);
        _crossNormBias = LongCatOps.WidenF32(w, $"{prefix}.pre_crs_attn_norm.bias", owned);
        _crossQ = w[$"{prefix}.cross_attn.q_linear.weight"];
        _crossQBias = LongCatOps.WidenF32(w, $"{prefix}.cross_attn.q_linear.bias", owned);
        _crossKv = w[$"{prefix}.cross_attn.kv_linear.weight"];
        _crossKvBias = LongCatOps.WidenF32(w, $"{prefix}.cross_attn.kv_linear.bias", owned);
        _crossQNorm = LongCatOps.WidenF32(w, $"{prefix}.cross_attn.q_norm.weight", owned);
        _crossKNorm = LongCatOps.WidenF32(w, $"{prefix}.cross_attn.k_norm.weight", owned);
        _crossProj = w[$"{prefix}.cross_attn.proj.weight"];
        _crossProjBias = LongCatOps.WidenF32(w, $"{prefix}.cross_attn.proj.bias", owned);
        _w1 = w[$"{prefix}.ffn.w1.weight"];
        _w2 = w[$"{prefix}.ffn.w2.weight"];
        _w3 = w[$"{prefix}.ffn.w3.weight"];
        BuildAttention();
        // Avatar blocks carry the audio step. It loads first, so the estimate below counts it. A base checkpoint has none of these keys: any step
        // from an earlier avatar load is released, so the block is a base block again.
        LongCatAvatarAudioStep? audio = w.ContainsKey($"{prefix}.audio_adaLN_modulation.1.weight")
            ? LongCatAvatarAudioStep.Load(w, prefix, _config.HiddenSize, _config.AdalnTembedDim, _config.AudioOutputDim, _config.NumHeads)
            : null;
        _audio?.Dispose();
        _audio = audio;
        long bytes = 0;
        foreach (Tensor t in EnumerateWeights())
            bytes += t.DType.ComputeByteCount(t.ElementCount);
        EstimatedWeightBytes = bytes;
    }

    /// <summary>Releases the avatar audio step's own tensors (the base block owns none). The block is unusable for a forward afterwards.</summary>
    public void Dispose()
    {
        _audio?.Dispose();
        _audio = null;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights() => OwnWeights();

    /// <summary>The adapter operands of this block's gated LoRA terms (each term's Down and Up). They are shared-resident, not streamed with the block:
    /// the transformer reports them with the shared weights, so they are preloaded once and held for the run. Row windows are left out: they are
    /// memoized slices that the backend promotes on use, and listing them would count the same operand twice.</summary>
    public IEnumerable<Tensor> EnumerateAdapterOperands()
    {
        foreach (Tensor weight in OwnWeights())
        {
            if (weight.LowRankAdjunct is null)
                continue;
            foreach (LowRankAdjunctTerm term in weight.LowRankAdjunct.Terms)
            {
                yield return term.Down;
                if (term.Up is not null)
                    yield return term.Up;
            }
        }
    }

    /// <summary>The block's own weights, without the adapter operands their LoRA terms carry.</summary>
    private IEnumerable<Tensor> OwnWeights()
    {
        foreach (Tensor? t in new[] { _adaWeight, _adaBias, _qkv, _qkvBias, _qNorm, _kNorm, _proj, _projBias, _crossNormWeight, _crossNormBias,
            _crossQ, _crossQBias, _crossKv, _crossKvBias, _crossQNorm, _crossKNorm, _crossProj, _crossProjBias, _w1, _w2, _w3 })
        {
            if (t is not null)
                yield return t;
        }
        if (_audio is not null)
            foreach (Tensor t in _audio.EnumerateWeights())
                yield return t;
    }


    private void BuildAttention()
    {
        AttentionLayerConfig self = new()
        {
            Heads = _config.NumHeads, KvHeads = _config.NumHeads, HeadDim = _config.HeadDim, Projection = AttentionProjection.FusedSplitNorm,
            QkNorm = AttentionQkNorm.PerHead, ComputeDType = DType.F32,
        };
        _selfAttention = new AttentionLayer(self, new AttentionLayerWeights
        {
            Qkv = _qkv, QkvBias = _qkvBias, Output = _proj, OutputBias = _projBias, QueryNorm = _qNorm, KeyNorm = _kNorm,
            QueryNormEps = LongCatConfig.NormEps, KeyNormEps = LongCatConfig.NormEps,
        });
        AttentionLayerConfig cross = self with { Projection = AttentionProjection.PerWeight };
        _crossAttention = new AttentionLayer(cross, new AttentionNormWeights(_crossQNorm, _crossKNorm, LongCatConfig.NormEps, LongCatConfig.NormEps));
    }

    /// <summary>One forward over the F32 residual stream <paramref name="hidden"/> <c>[S, dim]</c>, which stays the caller's.
    /// <paramref name="text"/> is the
    /// projected caption <c>[L, dim]</c>; <paramref name="timeEmbed"/> is the per-frame timestep embedding <c>[frames, adaln]</c>.
    /// <paramref name="probe"/> receives named intermediate stages when set. Returns a new <c>[S, dim]</c> stream.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor text, Tensor timeEmbed, in LongCatBlockInputs inputs,
        Action<string, Tensor>? probe = null, string stage = "", Tensor? audio = null) =>
        Forward(backend, hidden, text, timeEmbed, in inputs, null, probe, stage, audio);

    /// <summary>The forward with <paramref name="past"/>, the cached conditioning rows of the denoising pass, in front of the rows' own keys.</summary>
    internal Tensor Forward(IBackend backend, Tensor hidden, Tensor text, Tensor timeEmbed, in LongCatBlockInputs inputs, LongCatPastKv? past,
        Action<string, Tensor>? probe, string stage, Tensor? audio, LongCatSpeakerPass? speakers = null)
    {
        int dim = _config.HiddenSize;
        int s = checked(inputs.Frames * inputs.TokensPerFrame);
        if (hidden.DType != DType.F32 || hidden.Shape.Rank != 2 || hidden.Shape[0] != s || hidden.Shape[1] != dim)
            throw new ArgumentException($"The LongCat stream is F32 [{s}, {dim}]; got {hidden.DType} {hidden.Shape}.", nameof(hidden));
        if (timeEmbed.Shape[0] != inputs.Frames)
            throw new ArgumentException($"The timestep embedding has {timeEmbed.Shape[0]} rows for {inputs.Frames} frames.", nameof(timeEmbed));
        int hw = inputs.TokensPerFrame;

        using Tensor modulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _adaWeight!, _adaBias);
        probe?.Invoke($"{stage}.modulation", modulation);

        // 1. self-attention: modulation columns are [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp].
        using Tensor normed1 = LongCatOps.NormModulate(backend, hidden, modulation, dim, 0, hw);
        probe?.Invoke($"{stage}.attn_in", normed1);
        // With a cache (the denoising pass), the rows attend to the cached conditioning keys in front of their own.
        // With several speakers the self-attention also yields the reference map the audio step positions its queries from.
        float[]? speakerMap = null;
        using Tensor attn = past is { } cached
            ? CachedSelfAttention(backend, normed1, inputs.Cos, inputs.Sin, in cached, speakers, inputs.TokensPerFrame, out speakerMap)
            : SelfAttention(backend, normed1, in inputs, probe, stage, speakers, out speakerMap);
        probe?.Invoke($"{stage}.self_attn", attn);
        using Tensor afterSelf = Residual(backend, LongCatOps.GatedResidual(backend, hidden, attn, modulation, 2 * dim, dim, inputs.FrameRowIndex));
        probe?.Invoke($"{stage}.after_self", afterSelf);

        // 2. text cross-attention, noisy rows only (conditioning rows get a zero update).
        using Tensor afterCross = Residual(backend, CrossAttentionResidual(backend, afterSelf, text, in inputs, probe, stage));
        probe?.Invoke($"{stage}.after_cross", afterCross);

        // 2b. avatar audio step (audio tokens given): the noisy rows take the gated audio cross-attention, the conditioning rows are unchanged.
        Tensor? afterAudio = null;
        if (audio is not null)
        {
            if (_audio is null)
                throw new InvalidOperationException("The block has no avatar audio weights, but audio tokens were given.");
            afterAudio = Residual(backend, speakers is null
                ? _audio.Apply(backend, afterCross, audio, timeEmbed, inputs.CondFrames, hw)
                : SpeakerAudio(backend, afterCross, audio, timeEmbed, inputs.CondFrames, hw, speakers, speakerMap!, probe, stage));
            probe?.Invoke($"{stage}.after_audio", afterAudio);
        }
        Tensor stream = afterAudio ?? afterCross;
        try
        {
            // 3. SwiGLU FFN.
            using Tensor normed3 = LongCatOps.NormModulate(backend, stream, modulation, 4 * dim, 3 * dim, hw);
            using Tensor ffn = FeedForward(backend, normed3);
            probe?.Invoke($"{stage}.ffn", ffn);
            Tensor output = Residual(backend, LongCatOps.GatedResidual(backend, stream, ffn, modulation, 5 * dim, dim, inputs.FrameRowIndex));
            probe?.Invoke($"{stage}.out", output);
            return output;
        }
        finally
        {
            afterAudio?.Dispose();
        }
    }

    /// <summary>Whether this block rounds its residual sums to BF16 (the reference's x dtype on a BF16 or INT8 release).</summary>
    internal bool Bf16Residual => _bf16Residual;

    /// <summary>The residual after an update: on a BF16 route the sum is rounded to BF16 (nearest-even) and kept as F32 values on the BF16 grid;
    /// otherwise it is returned unchanged. The sum is disposed when it is rounded.</summary>
    private Tensor Residual(IBackend backend, Tensor sum)
    {
        if (!_bf16Residual)
            return sum;
        Tensor rounded = LongCatOps.RoundToBf16(backend, sum);
        sum.Dispose();
        return rounded;
    }

    private int _qkvTileRows = DefaultTileRows;

    /// <summary>Sets the QKV projection tile height. A value at or above the token count projects untiled; tests use small tiles to exercise the seams.</summary>
    internal void SetQkvTileRows(int rows)
    {
        if (rows < 1) throw new ArgumentOutOfRangeException(nameof(rows));
        _qkvTileRows = rows;
    }

    /// <summary>The self-attention projection over row tiles. The projection is row-wise (a linear, then the per-head QK norm), so the joined Q, K and
    /// V equal the untiled projection; the rotation and the attention run after, on the joined tensors.</summary>
    private (Tensor Q, Tensor K, Tensor V) ProjectSelfTiled(IBackend backend, Tensor input)
    {
        int s = (int)input.Shape[0];
        int dim = (int)input.Shape[1];
        int tile = Math.Min(_qkvTileRows, s);
        if (tile >= s)
            return _selfAttention!.ProjectSelf(backend, input, releaseInput: false);

        List<Tensor> qs = new((s + tile - 1) / tile), ks = new(qs.Capacity), vs = new(qs.Capacity);
        Tensor? joinedQ = null, joinedK = null;
        try
        {
            for (int start = 0; start < s; start += tile)
            {
                int rows = Math.Min(tile, s - start);
                using Tensor chunk = SliceRows(backend, input, start, rows, dim);
                (Tensor q, Tensor k, Tensor v) = _selfAttention!.ProjectSelf(backend, chunk, releaseInput: false);
                qs.Add(q);
                ks.Add(k);
                vs.Add(v);
            }
            // Each group's tiles are freed as soon as that group is joined, so at the last join the peak is two joined outputs plus the
            // last group's tiles and its output, not every tile and every output at once.
            joinedQ = JoinRows(backend, qs, s);
            DisposeAll(qs);
            joinedK = JoinRows(backend, ks, s);
            DisposeAll(ks);
            Tensor joinedV = JoinRows(backend, vs, s);
            DisposeAll(vs);
            return (joinedQ, joinedK, joinedV);
        }
        catch
        {
            joinedQ?.Dispose();
            joinedK?.Dispose();
            throw;
        }
        finally
        {
            DisposeAll(qs);
            DisposeAll(ks);
            DisposeAll(vs);
        }
    }

    private static void DisposeAll(List<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.Dispose();
        tensors.Clear();
    }

    /// <summary>Concatenates row tiles of one projection into a single [rows, ...] tensor of the tiles' dtype.</summary>
    private static Tensor JoinRows(IBackend backend, List<Tensor> parts, int rows)
    {
        Tensor first = parts[0];
        TensorShape shape = first.Shape.Rank switch
        {
            2 => new TensorShape(rows, first.Shape[1]),
            3 => new TensorShape(rows, first.Shape[1], first.Shape[2]),
            _ => throw new NotSupportedException($"A tiled LongCat projection joins rank-2 or rank-3 tiles; got rank {first.Shape.Rank}."),
        };
        Tensor joined = new(shape, first.DType);
        try
        {
            backend.Concat(joined, parts.ToArray(), 0);
            return joined;
        }
        catch
        {
            joined.Dispose();
            throw;
        }
    }

    private Tensor SelfAttention(IBackend backend, Tensor input, in LongCatBlockInputs inputs, Action<string, Tensor>? probe, string stage,
        LongCatSpeakerPass? speakers, out float[]? speakerMap)
    {
        speakerMap = null;
        int s = (int)input.Shape[0];
        QkRotation rotation = new(QkRotationKind.InterleavedPairs, inputs.Cos, inputs.Sin, s);
        AttentionCallContext context = new() { QueryRotation = rotation, KeyRotation = rotation };
        if (inputs.Segments.Length == 1)
        {
            AttentionSegment.Validate(inputs.Segments, s);
            // The fused path needs one projection tile, and it exposes no rotated queries and keys for the reference map.
            if (speakers is null && inputs.Segments[0].KeyCount == s && s <= _qkvTileRows)
                return _selfAttention!.Forward(backend, input, input, in context);
        }

        (Tensor q, Tensor k, Tensor v) = ProjectSelfTiled(backend, input);
        try
        {
            _selfAttention!.RotateQk(backend, q, k, in context);
            if (probe is not null)
            {
                probe($"{stage}.q", q);
                probe($"{stage}.k", k);
                probe($"{stage}.v", v);
            }
            if (speakers is not null)
            {
                // The noisy rows' queries against the first latent frame's keys.
                using Tensor referenceKeys = HeadMajorFirstRows(backend, k, inputs.TokensPerFrame);
                speakerMap = speakers.ReferenceMap(backend, _selfAttention, q, inputs.CondFrames * inputs.TokensPerFrame, referenceKeys);
            }
            Tensor flat = SegmentedAttention.Run(backend, _selfAttention, q, k, v, inputs.Segments);
            return _selfAttention.ProjectOutput(backend, flat);
        }
        finally
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
        }
    }

    /// <summary><c>x + cross_attn(norm(x))</c> for the rows past the conditioning frames; the conditioning rows pass through.</summary>
    private Tensor CrossAttentionResidual(IBackend backend, Tensor x, Tensor text, in LongCatBlockInputs inputs, Action<string, Tensor>? probe,
        string stage)
    {
        int dim = _config.HiddenSize;
        int s = (int)x.Shape[0];
        int condRows = checked(inputs.CondFrames * inputs.TokensPerFrame);
        int noisyRows = s - condRows;
        if (noisyRows <= 0)
            return LongCatOps.Copy(backend, x);

        using Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _crossNormWeight!, _crossNormBias!, LongCatConfig.NormEps);
        using Tensor? noisyRowsInput = condRows == 0 ? null : SliceRows(backend, normed, condRows, noisyRows, dim);
        using Tensor update = CrossAttention(backend, noisyRowsInput ?? normed, text);
        probe?.Invoke($"{stage}.cross_attn", update);
        if (condRows == 0)
            return LongCatOps.Add(backend, x, update);

        // Conditioning rows pass through untouched: only the noisy rows take the update, joined back on device.
        using Tensor conditioning = SliceRows(backend, x, 0, condRows, dim);
        using Tensor noisy = SliceRows(backend, x, condRows, noisyRows, dim);
        using Tensor updatedNoisy = LongCatOps.Add(backend, noisy, update);
        Tensor joined = new(new TensorShape(s, dim), DType.F32);
        backend.Concat(joined, [conditioning, updatedNoisy], 0);
        return joined;
    }

    private Tensor CrossAttention(IBackend backend, Tensor queryInput, Tensor text)
    {
        int heads = _config.NumHeads, headDim = _config.HeadDim, dim = _config.HiddenSize;
        int rows = (int)queryInput.Shape[0], textRows = (int)text.Shape[0];
        Tensor q = new(new TensorShape(1, rows, heads, headDim), DType.F32);
        Tensor k = new(new TensorShape(1, textRows, heads, headDim), DType.F32);
        Tensor v = new(new TensorShape(1, textRows, heads, headDim), DType.F32);
        try
        {
            backend.Linear(q, queryInput, _crossQ!, _crossQBias);
            // kv_linear is one [2·dim, dim] weight: k is its first dim rows, v the next; windowed so a LoRA adjunct on it applies per part.
            backend.LinearWeightRows(k, text, _crossKv!, _crossKvBias, 0, dim);
            backend.LinearWeightRows(v, text, _crossKv!, _crossKvBias, dim, dim);
        }
        catch
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
        // Once its argument checks pass, Attend owns q, k and v and disposes them, even if it throws later; so it sits outside the catch above.
        Tensor flat = _crossAttention!.Attend(backend, q, k, v, new TensorShape(rows, dim), default);
        using (flat)
        {
            Tensor projected = new(new TensorShape(rows, dim), DType.F32);
            backend.Linear(projected, flat, _crossProj!, _crossProjBias);
            return projected;
        }
    }

    /// <summary>The SwiGLU FFN over row tiles: each tile's intermediates are freed before the next, so the peak is one tile's FFN plus the output.</summary>
    private Tensor FeedForward(IBackend backend, Tensor x)
    {
        int s = (int)x.Shape[0];
        int dim = _config.HiddenSize;
        int tile = Math.Min(_ffnTileRows, s);
        if (tile >= s)
            return FeedForwardRows(backend, x);

        List<Tensor> parts = new((s + tile - 1) / tile);
        try
        {
            for (int start = 0; start < s; start += tile)
            {
                int rows = Math.Min(tile, s - start);
                using Tensor chunk = SliceRows(backend, x, start, rows, dim);
                parts.Add(FeedForwardRows(backend, chunk));
            }
            Tensor output = new(new TensorShape(s, dim), DType.F32);
            try
            {
                backend.Concat(output, parts.ToArray(), 0);
            }
            catch
            {
                output.Dispose();
                throw;
            }
            return output;
        }
        finally
        {
            foreach (Tensor part in parts)
                part.Dispose();
        }
    }

    private Tensor FeedForwardRows(IBackend backend, Tensor x)
    {
        int s = (int)x.Shape[0];
        int hidden = _config.FfnHiddenDim;
        using Tensor gate = new(new TensorShape(s, hidden), DType.F32);
        backend.Linear(gate, x, _w1!, null);
        using Tensor activated = new(gate.Shape, DType.F32);
        backend.Silu(activated, gate);
        using Tensor up = new(gate.Shape, DType.F32);
        backend.Linear(up, x, _w3!, null);
        using Tensor product = new(gate.Shape, DType.F32);
        backend.Mul(product, activated, up);
        Tensor output = new(new TensorShape(s, _config.HiddenSize), DType.F32);
        backend.Linear(output, product, _w2!, null);
        return output;
    }

    private static Tensor SliceRows(IBackend backend, Tensor source, int start, int count, int dim)
    {
        Tensor rows = new(new TensorShape(count, dim), DType.F32);
        backend.SliceRows(rows, source, start);
        return rows;
    }
}
