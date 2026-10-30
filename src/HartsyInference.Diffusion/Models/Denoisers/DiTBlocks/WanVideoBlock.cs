using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Video DiT block (<c>WanTransformerBlock</c>), ported from diffusers. B=1 over <c>[S, dim]</c>: self-attention + per-head 3D RoPE → cross-attention to umT5 → gelu-approx FFN. AdaLN is 6-param (self-attn shift/scale/gate + FFN shift/scale/gate; cross-attn ungated). Pre-norms are FP32 LayerNorm (norm1/norm3 no-affine, norm2 affine when cross_attn_norm); QK-norm is RMSNorm-across-heads. Reuses backend <c>LayerNorm</c>/<c>RmsNorm</c>/<c>ScaledDotProductAttention</c> + <see cref="WanRope"/>.</summary>
public sealed unsafe class WanVideoBlock : IStreamingBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly float _eps;
    private readonly bool _crossAttnNorm;

    private Tensor? _scaleShift;        // [6, dim]
    private Tensor? _norm2W, _norm2B;   // FP32LayerNorm affine (cross), when cross_attn_norm
    // attn1 (self) + attn2 (cross)
    private Tensor?[] _q = new Tensor?[2], _qB = new Tensor?[2], _k = new Tensor?[2], _kB = new Tensor?[2],
        _v = new Tensor?[2], _vB = new Tensor?[2], _o = new Tensor?[2], _oB = new Tensor?[2], _nq = new Tensor?[2], _nk = new Tensor?[2];
    private Tensor? _ffProjW, _ffProjB, _ffOutW, _ffOutB;
    // I2V image cross-attention KV (cross-attn to CLIP image context); present only when the checkpoint ships them.
    private Tensor? _addK, _addKB, _addV, _addVB, _normAddedK;
    private AttentionLayer? _selfAttention, _crossAttention, _imageAttention, _selfAttention16, _crossAttention16;

    internal AttentionLayer? SelfAttentionStage => _selfAttention;
    internal AttentionLayer? CrossAttentionStage => _crossAttention;
    internal AttentionLayer? ImageAttentionStage => _imageAttention;
    internal AttentionLayer? SelfAttentionF16Stage => _selfAttention16;
    internal AttentionLayer? CrossAttentionF16Stage => _crossAttention16;

    public WanVideoBlock(WanVideoConfig c, bool crossAttnNorm)
    {
        _dim = c.InnerDim;
        _heads = c.NumHeads;
        _headDim = c.HeadDim;
        _eps = c.Eps;
        _crossAttnNorm = crossAttnNorm;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _scaleShift = TensorCasts.LoadF32(w, $"{prefix}.scale_shift_table");   // [1,6,dim] → flat 6*dim
        LoadAttn(w, $"{prefix}.attn1", 0);
        LoadAttn(w, $"{prefix}.attn2", 1);
        // I2V image cross-attention (only present in image-conditioned checkpoints).
        if (w.TryGetValue($"{prefix}.attn2.add_k_proj.weight", out Tensor? addK))
        {
            _addK = addK; w.TryGetValue($"{prefix}.attn2.add_k_proj.bias", out _addKB);
            _addV = w[$"{prefix}.attn2.add_v_proj.weight"]; w.TryGetValue($"{prefix}.attn2.add_v_proj.bias", out _addVB);
            _normAddedK = TensorCasts.LoadF32(w, $"{prefix}.attn2.norm_added_k.weight");
        }
        // norm2 affine is read by the manual host-pointer LayerNorm loop → cast BOTH weight and bias to F32
        // (bf16 checkpoints else feed garbage bias bytes reinterpreted as f32).
        if (_crossAttnNorm) { _norm2W = TensorCasts.LoadF32(w, $"{prefix}.norm2.weight"); _norm2B = w.TryGetValue($"{prefix}.norm2.bias", out Tensor? n2b) ? (n2b.DType == DType.F32 ? n2b : n2b.CastTo(DType.F32)) : null; }
        _ffProjW = w[$"{prefix}.ffn.net.0.proj.weight"]; w.TryGetValue($"{prefix}.ffn.net.0.proj.bias", out _ffProjB);
        _ffOutW = w[$"{prefix}.ffn.net.2.weight"]; w.TryGetValue($"{prefix}.ffn.net.2.bias", out _ffOutB);
        long bytes = 0;
        foreach (Tensor t in EnumerateWeights()) bytes += t.DType.ComputeByteCount(t.ElementCount);
        EstimatedWeightBytes = bytes;
        BuildAttention();
    }

    private void BuildAttention()
    {
        // allowF16: Q and K are RMS-normed, so the scores are bounded.
        AttentionLayerConfig config = new()
        {
            Heads = _heads, KvHeads = _heads, HeadDim = _headDim, QkNorm = AttentionQkNorm.FullWidthRms, ComputeDType = DType.F32,
            AllowF16 = true,
        };
        // F16 is named rather than followed from the input: the text context the cross-attention K/V project stays F32.
        // The F16 branches were measured on cuDNN's F16 attention, so they refuse SageAttention's INT8.
        AttentionLayerConfig config16 = config with { ComputeDType = DType.F16, RefusedFormats = AttentionFormats.Int8 };
        AttentionLayerWeights Weights(int i) => new()
        {
            Query = _q[i], QueryBias = _qB[i], Key = _k[i], KeyBias = _kB[i], Value = _v[i], ValueBias = _vB[i], Output = _o[i],
            OutputBias = _oB[i], QueryNorm = _nq[i], KeyNorm = _nk[i], QueryNormEps = _eps, KeyNormEps = _eps,
        };
        static AttentionLayerConfig TokenMajor(AttentionLayerConfig c) =>
            c with { TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = true };
        _selfAttention = new AttentionLayer(TokenMajor(config), Weights(0));
        _crossAttention = new AttentionLayer(config, Weights(1));
        _selfAttention16 = new AttentionLayer(TokenMajor(config16), Weights(0));
        _crossAttention16 = new AttentionLayer(config16, Weights(1));
        _imageAttention = _addK is null ? null : new AttentionLayer(config, new AttentionLayerWeights
        {
            Key = _addK, KeyBias = _addKB, Value = _addV, ValueBias = _addVB, KeyNorm = _normAddedK, KeyNormEps = _eps,
        });
    }

    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c> — the latter reports
    /// 0 for block-quantized dtypes, which would size a streaming window at zero.</remarks>
    public long EstimatedWeightBytes { get; private set; }

    private void LoadAttn(IReadOnlyDictionary<string, Tensor> w, string p, int i)
    {
        _q[i] = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out Tensor? qb); _qB[i] = qb;
        _k[i] = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out Tensor? kb); _kB[i] = kb;
        _v[i] = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out Tensor? vb); _vB[i] = vb;
        _o[i] = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out Tensor? ob); _oB[i] = ob;
        _nq[i] = TensorCasts.LoadF32(w, $"{p}.norm_q.weight");
        _nk[i] = TensorCasts.LoadF32(w, $"{p}.norm_k.weight");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scaleShift is not null) yield return _scaleShift;
        if (_norm2W is not null) yield return _norm2W;
        if (_norm2B is not null) yield return _norm2B;
        for (int i = 0; i < 2; i++)
            foreach (Tensor? t in new[] { _q[i], _qB[i], _k[i], _kB[i], _v[i], _vB[i], _o[i], _oB[i], _nq[i], _nk[i] })
                if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _ffProjW, _ffProjB, _ffOutW, _ffOutB }) if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _addK, _addKB, _addV, _addVB, _normAddedK }) if (t is not null) yield return t;
    }

    /// <summary>Forward over <c>[S, dim]</c>. <paramref name="temb"/> is the per-block modulation <c>[G, 6, dim]</c>
    /// (timestep_proj); the block adds its <c>scale_shift_table</c>. G=1 is the standard scalar-timestep path;
    /// G=latent-frames with <paramref name="tokensPerGroup"/> tokens each is the TI2V per-frame-timestep path
    /// (tokens are in (t,h,w) order, so each frame's tokens are contiguous). <paramref name="postCrossAttnHook"/>
    /// runs between the cross-attention residual and the FFN — Matrix-Game attaches its ActionModule there (the hook
    /// mutates the hidden state in place). <paramref name="selfAttnMask"/> is an optional additive mask broadcastable
    /// to <c>[1, heads, S, S]</c> — Matrix-Game 2's block-causal + local-window attention.
    /// <paramref name="animate2"/> replaces self-attention with Wan-Animate-2's frame-local splice against the
    /// driving stream's re-projected K/V; <paramref name="selfAttnKvCapture"/> hands the caller this block's
    /// self-attention input (the Animate-2 driving prepass, which is otherwise an ordinary block forward).
    /// <paramref name="selfAttnKvRewrite"/> trades the local K/V for the full sequence's (context parallel).
    /// <paramref name="textKv"/> keeps the text cross-attention K/V across forwards; it is used only without an image context.
    /// <para>The residual stream <paramref name="hidden"/> is F32. <paramref name="branchDType"/> F16 runs the attention, the
    /// cross-attention and the FFN in F16 for the plain single-group forward: each norm writes its branch input in F16 and each
    /// branch output is added back to the F32 stream; the modulation, the norm weights and the RoPE tables stay F32.</para></summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoder, Tensor temb, WanRope rope, Tensor cos, Tensor sin, int tokensPerGroup,
        Action<Tensor>? postCrossAttnHook = null, Tensor? selfAttnMask = null, int imageContextLen = 0, string? dbg = null,
        Action<Tensor>? postSelfAttnHook = null, IKvRewrite? selfAttnKvRewrite = null,
        WanAnimate2SelfAttnContext? animate2 = null, WanAnimate2KvCapture? selfAttnKvCapture = null, ICachedKv? textKv = null,
        DType? branchDType = null)
    {
        int s = (int)hidden.Shape[0];
        DType branch = branchDType ?? DType.F32;
        if (hidden.DType != DType.F32)
            throw new NotSupportedException($"The Wan residual stream is F32, got {hidden.DType}.");
        // A K/V rewrite is allowed: context parallel's exchange is the same rewrite for F16 rows as for F32.
        if (branch != DType.F32 && (branch != DType.F16 || tokensPerGroup != s || temb.Shape[0] != 1 || imageContextLen != 0
            || postCrossAttnHook is not null || selfAttnMask is not null || postSelfAttnHook is not null
            || animate2 is not null || selfAttnKvCapture is not null || CrossAttnResidualNormed))
            throw new NotSupportedException(
                $"{branch} Wan branches run one timestep group with no image context, mask, hook or Animate-2 splice.");
        bool f16Branches = branch == DType.F16;
        // The dump wants the raw scale rows, and so does the fused F16 norm, which adds the 1 itself.
        bool scalePlusOne = !f16Branches && tokensPerGroup == s && temb.Shape[0] == 1 && !WanVideoDebugDump.Enabled;
        (Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa, Tensor cShift, Tensor cScale, Tensor cGate) = Modulation(backend, temb, scalePlusOne);
        if (dbg != null) { WanVideoDebugDump.Dump($"{dbg}_scaleMsa", scaleMsa); WanVideoDebugDump.Dump($"{dbg}_gateMsa", gateMsa); }

        // 1. self-attn
        Tensor n1 = f16Branches ? NormModulateF16(backend, hidden, scaleMsa, shiftMsa, s)
            : ApplyShiftScale(backend, LayerNorm(backend, hidden, null, null, s, DType.F32), scaleMsa, shiftMsa, s, tokensPerGroup, scalePlusOne);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_n1", n1);
        Tensor attn1 = SelfAttention(backend, n1, cos, sin, s, selfAttnMask, selfAttnKvRewrite, animate2, selfAttnKvCapture);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_attn1", attn1);
        n1.Dispose();
        Tensor h1 = GatedAdd(backend, hidden, attn1, gateMsa, s, tokensPerGroup);
        attn1.Dispose();

        // Optional post-self-attn injection (Matrix-Game 3.0's per-block Plücker camera modulation
        // `x = (1+cam_scale)·x + cam_shift`, applied between the self-attn residual and cross-attn); mutates h1 in place.
        postSelfAttnHook?.Invoke(h1);

        // 2. cross-attn (to umT5 text, plus optional CLIP image context for I2V)
        Tensor n2 = LayerNorm(backend, h1, _norm2W, _norm2B, s, branch);
        Tensor attn2 = CrossAttention(backend, n2, encoder, imageContextLen, imageContextLen == 0 ? textKv : null);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_attn2", attn2);
        // Matrix-Game 3.0's memory-mode block builds the cross-attn residual on the NORMED hidden state
        // (`x = norm3(x); x = x + cross_attn(x)` in wan/modules/model.py cross_attn_ffn), so norm3 is destructive.
        // Default (whole video fleet) keeps the standard `x + cross_attn(norm3(x))` residual on the un-normed h1.
        Tensor h2 = CrossAttnResidualNormed ? AddRows(backend, n2, attn2, s) : AddRows(backend, h1, attn2, s);
        n2.Dispose();
        h1.Dispose();
        attn2.Dispose();

        postCrossAttnHook?.Invoke(h2);

        // 3. ffn
        Tensor n3 = f16Branches ? NormModulateF16(backend, h2, cScale, cShift, s)
            : ApplyShiftScale(backend, LayerNorm(backend, h2, null, null, s, DType.F32), cScale, cShift, s, tokensPerGroup, scalePlusOne);
        Tensor ff = Ffn(backend, n3, s);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_ff", ff);
        n3.Dispose();
        Tensor outT = GatedAdd(backend, h2, ff, cGate, s, tokensPerGroup);
        h2.Dispose();
        ff.Dispose();

        foreach (Tensor t in new[] { shiftMsa, scaleMsa, gateMsa, cShift, cScale, cGate }) t.Dispose();
        return outT;
    }

    private Tensor SelfAttention(IBackend backend, Tensor input, Tensor cos, Tensor sin, int s, Tensor? mask, IKvRewrite? kvRewrite,
        WanAnimate2SelfAttnContext? animate2, WanAnimate2KvCapture? kvCapture)
    {
        // Rank-2 cos is shared across heads; the rank-3 per-head sigma_theta table needs its own kernel.
        QkRotation rotation = new(cos.Shape.Rank == 2 ? QkRotationKind.InterleavedPairs : QkRotationKind.InterleavedPairsPerHead, cos, sin, s);
        IQkvTap? tap = kvCapture is null ? null : new Animate2Tap(this, kvCapture, input, s);
        AttentionCallContext context = new()
        {
            QueryRotation = rotation, KeyRotation = rotation, Bias = mask, Tap = tap, KvRewrite = kvRewrite,
        };
        if (animate2 is null)
        {
            if (input.DType == DType.F16)
                return _selfAttention16!.Forward(backend, input, input, in context);
            if (tap is not null || !LinearsCastInputToF16(backend, _q[0]!, _k[0]!, _v[0]!))
                return _selfAttention!.Forward(backend, input, input, in context);
            // One cast for the three projections, each of which would cast its F32 input with the same kernel.
            using Tensor input16 = new(input.Shape, DType.F16);
            backend.CastToF16(input16, input);
            return _selfAttention!.Forward(backend, input16, input16, in context);
        }

        (Tensor qn, Tensor kn, Tensor v) = _selfAttention!.ProjectSelf(backend, input, releaseInput: false);
        _selfAttention.NormalizeQk(backend, ref qn, ref kn);
        tap?.Observe(backend, qn, kn, v);
        _selfAttention.RotateQk(backend, qn, kn, in context);
        Tensor spliced = Animate2FrameLocalAttention(backend, qn, kn, v, s, 0, animate2);
        qn.Dispose(); kn.Dispose(); v.Dispose();
        return _selfAttention.ProjectOutput(backend, spliced);
    }

    /// <summary>True when an F16 input reaches the same GEMM as the F32 input these weights' Linears would cast to F16 themselves:
    /// plain F16 weights on a backend whose F16 activations take the F16 GEMM, without the W8A8 path, which quantizes its own input.</summary>
    private static bool LinearsCastInputToF16(IBackend backend, params ReadOnlySpan<Tensor> weights)
    {
        if (!backend.SupportsF16Activations || EngineKnobs.W8a8.Value)
            return false;
        foreach (Tensor w in weights)
            if (w.DType != DType.F16 || w.LowRankAdjunct is { IsActive: true })
                return false;
        return true;
    }

    /// <summary>Clones the Animate-2 prepass's block input and its post-norm, pre-RoPE K and V, so the generation stream re-projects and rotates on read.</summary>
    private sealed class Animate2Tap(WanVideoBlock block, WanAnimate2KvCapture capture, Tensor input, int rows) : IQkvTap
    {
        public void Observe(IBackend backend, Tensor q, Tensor k, Tensor v)
        {
            capture.Input = block.CloneRows(backend, input, rows);
            if (capture.CaptureProjected)
            {
                capture.K = block.CloneRows(backend, k, rows);
                capture.V = block.CloneRows(backend, v, rows);
            }
        }
    }

    /// <summary>Wan-Animate-2 frame-local self-attention: one attention call per generation latent frame, over all
    /// generation tokens plus (for frames &gt; 0) the single aligned driving frame. Ported from ComfyUI's loop form
    /// rather than the official flex-attention kernel, whose <c>ceil(/128)·128</c> and <c>sp_size</c> padding are
    /// kernel artifacts. The cache holds the driving stream's block INPUT, so K and V are re-projected here and the
    /// keys are rotated with the driving RoPE table afterwards — still pre-RoPE storage, still per-call offsets.
    /// <paramref name="qn"/>/<paramref name="kn"/> are already rotated with the generation table.</summary>
    private Tensor Animate2FrameLocalAttention(IBackend backend, Tensor qn, Tensor kn, Tensor v, int s, int idx,
        WanAnimate2SelfAttnContext ctx)
    {
        int hw = ctx.TokensPerFrame, frames = ctx.GenFrames;
        int refSeq = (frames - 1) * hw;
        if (s != frames * hw)
            throw new ArgumentException($"Animate-2 sequence {s} != {frames} frames × {hw} tokens.", nameof(s));
        Tensor stored = ctx.DrivingInput ?? throw new InvalidOperationException("Animate-2 block has no cached driving input.");
        if (stored.Shape[0] != refSeq)
            throw new ArgumentException($"Driving cache holds {stored.Shape[0]} tokens; expected {refSeq} ({frames - 1} frames × {hw}).");

        Tensor drivingInput = stored;
        if (stored.DType != DType.F32)
        {
            drivingInput = new Tensor(stored.Shape, DType.F32);
            backend.CastToF32(drivingInput, stored);
        }
        (Tensor kRef, Tensor drivingV) = ProjectDrivingKv(backend, drivingInput, refSeq, idx);
        if (!ReferenceEquals(drivingInput, stored)) drivingInput.Dispose();
        if (ctx.PoseStrength != 1.0f) backend.Scale(drivingV, drivingV, ctx.PoseStrength);
        backend.WanRopeInterleaved(kRef, ctx.RefCos, ctx.RefSin, refSeq, _heads, _headDim);

        ScaleReferenceRows(backend, v, hw, _dim, ctx.ReferenceImageStrength);
        Tensor kGen = ToBhsd(backend, kn, s);
        Tensor vGen = ToBhsd(backend, v, s);
        backend.ScatterSeqHeadMajor(ctx.KeyBuffer, kGen, 0);
        backend.ScatterSeqHeadMajor(ctx.ValueBuffer, vGen, 0);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor outMh = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        for (int j = 0; j < frames; j++)
        {
            using Tensor qRows = new Tensor(new TensorShape(hw, _dim), DType.F32);
            backend.SliceRows(qRows, qn, j * hw);
            Tensor qMh = ToBhsd(backend, qRows, hw);

            Tensor keys = kGen, values = vGen;
            Tensor? bias = ctx.LogScaleBiasGen;
            if (j > 0)
            {
                ScatterDrivingFrame(backend, ctx.KeyBuffer, kRef, j - 1, hw, s);
                ScatterDrivingFrame(backend, ctx.ValueBuffer, drivingV, j - 1, hw, s);
                keys = ctx.KeyBuffer; values = ctx.ValueBuffer;
                bias = ctx.LogScaleBiasSpliced;
            }
            Tensor attn = new Tensor(new TensorShape(1, _heads, hw, _headDim), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, keys, values, bias, scale, allowF16: true);
            qMh.Dispose();
            backend.ScatterSeqHeadMajor(outMh, attn, j * hw);
            attn.Dispose();
        }
        kRef.Dispose(); drivingV.Dispose(); kGen.Dispose(); vGen.Dispose();
        Tensor flat = FromBhsd(backend, outMh, s);
        outMh.Dispose();
        return flat;
    }

    /// <summary>Re-derives the driving stream's pre-RoPE K and V from its cached block input — the same
    /// <c>Linear → RmsNorm</c> for K and <c>Linear</c> for V the prepass ran, so the pair is numerically the pair the
    /// prepass would have stored.</summary>
    internal (Tensor K, Tensor V) ProjectDrivingKv(IBackend backend, Tensor input, int refSeq, int idx)
    {
        Tensor k = new Tensor(new TensorShape(refSeq, _dim), DType.F32);
        backend.Linear(k, input, _k[idx]!, _kB[idx]);
        Tensor kn = new Tensor(k.Shape, DType.F32);
        backend.RmsNorm(kn, k, _nk[idx]!, _eps);
        k.Dispose();
        Tensor v = new Tensor(new TensorShape(refSeq, _dim), DType.F32);
        backend.Linear(v, input, _v[idx]!, _vB[idx]);
        return (kn, v);
    }

    /// <summary>ComfyUI Animate-2's <c>v[:, :hw] *= ref_strength</c>: scales the reference-image slot's V rows
    /// <c>[0, hw)</c> in place, leaving every other row untouched; exact no-op at 1.0.</summary>
    internal static void ScaleReferenceRows(IBackend backend, Tensor v, int hw, int dim, float strength)
    {
        if (strength == 1.0f) return;
        using Tensor rows = new Tensor(new TensorShape(hw, dim), DType.F32);
        backend.SliceRows(rows, v, 0);
        backend.Scale(rows, rows, strength);
        backend.ScatterRowsGeneric(v, rows, 0);
    }

    /// <summary>Writes driving frame <paramref name="frame"/>'s <c>hw</c> tokens into the splice buffer's tail rows.</summary>
    private void ScatterDrivingFrame(IBackend backend, Tensor buffer, Tensor source, int frame, int hw, int tailOffset)
    {
        using Tensor rows = new Tensor(new TensorShape(hw, _dim), DType.F32);
        backend.SliceRows(rows, source, frame * hw);
        using Tensor mh = ToBhsd(backend, rows, hw);
        backend.ScatterSeqHeadMajor(buffer, mh, tailOffset);
    }

    /// <summary>Device-resident row copy — a host <c>Buffer.MemoryCopy</c> here would drag the tensor off the GPU.</summary>
    private Tensor CloneRows(IBackend backend, Tensor x, int rows)
    {
        Tensor o = new Tensor(new TensorShape(rows, _dim), DType.F32);
        backend.SliceRows(o, x, 0);
        return o;
    }

    /// <summary>Cross-attention to the umT5 text context, with an optional CLIP image branch (I2V): the first
    /// <paramref name="imageContextLen"/> encoder rows are the projected image context (attended via
    /// <c>add_k_proj</c>/<c>add_v_proj</c>), the rest are text. Both branches share the query; their per-head outputs
    /// are summed before the output projection (matching diffusers' WanAttnProcessor).</summary>
    private Tensor CrossAttention(IBackend backend, Tensor qInput, Tensor encoder, int imageContextLen, ICachedKv? textKv)
    {
        int sq = (int)qInput.Shape[0];
        int l = (int)encoder.Shape[0];
        int textLen = l - imageContextLen;
        TensorShape flatShape = new(sq, _dim);
        AttentionLayer main = qInput.DType == DType.F16 ? _crossAttention16! : _crossAttention!;

        Tensor qMh = main.ProjectQueryHeadMajor(backend, qInput, releaseInput: false, default);

        Tensor textRows = imageContextLen > 0 ? SliceRows(backend, encoder, imageContextLen, textLen) : encoder;
        Tensor flat = main.MergeHeads(backend, main.AttendBranchHeadMajor(backend, qMh, textRows, new AttentionCallContext { CachedKv = textKv }), flatShape);
        if (imageContextLen > 0) textRows.Dispose();

        if (imageContextLen > 0 && _imageAttention is { } image)
        {
            Tensor imgRows = SliceRows(backend, encoder, 0, imageContextLen);
            Tensor flatImg = image.MergeHeads(backend, image.AttendBranchHeadMajor(backend, qMh, imgRows, default), flatShape);
            imgRows.Dispose();
            // Device add — the old host AddInPlace pointer-loop drained BOTH [sq, dim] branch outputs D2H, summed on
            // the CPU, and re-uploaded on next use: ~0.5 GB of synchronous PCIe round-trips per block per forward
            // (×40 blocks ×2 CFG = the dominant Wan2.1-CLIP-I2V cost over T2V).
            Tensor summed = AddRows(backend, flat, flatImg, sq);
            flat.Dispose();
            flatImg.Dispose();
            flat = summed;
        }
        qMh.Dispose();

        return main.ProjectOutput(backend, flat);
    }

    /// <summary>Device row slice of the encoder context — a host <c>Buffer.MemoryCopy</c> here produced a FRESH
    /// host tensor per block per forward whose first device use was a cache MISS (full stream drain + pageable H2D,
    /// the sync-H2D disease).</summary>
    private Tensor SliceRows(IBackend backend, Tensor x, int start, int len)
    {
        Tensor o = new Tensor(new TensorShape(len, _dim), DType.F32);
        backend.SliceRows(o, x, start);
        return o;
    }

    /// <summary>Activation dtype for an F32 stream's FFN [s, ffn_dim] intermediate — <c>DType.F16</c> halves its
    /// bandwidth (the dominant per-block activation), F32-accumulated in the GEMMs, tiny cast boundaries. Default
    /// F32 keeps every existing caller (the whole video fleet) byte-identical; opt-in models (Matrix-Game 2.0) set
    /// it to <see cref="DiTBlocks.DitDtype.Act"/>. An F16 stream's FFN is F16 regardless.</summary>
    internal DType FfnDtype = DType.F32;

    /// <summary>When true, the cross-attention residual is built on the NORMED hidden state (Matrix-Game 3.0's
    /// memory-mode block: <c>x = norm3(x); x = x + cross_attn(x)</c>). Default false = the standard Wan
    /// <c>x + cross_attn(norm3(x))</c> residual (the whole video fleet). Opt-in per model; no video regression.</summary>
    internal bool CrossAttnResidualNormed = false;

    private Tensor Ffn(IBackend backend, Tensor x, int s)
    {
        int inner = (int)_ffProjW!.Shape[0];
        if (x.DType == DType.F32 && FfnDtype == DType.F16 && backend.SupportsF16Activations)
        {
            Tensor xF16 = new Tensor(x.Shape, DType.F16); backend.CastToF16(xF16, x);
            Tensor projH = new Tensor(new TensorShape(s, inner), DType.F16); backend.Linear(projH, xF16, _ffProjW!, _ffProjB);
            xF16.Dispose();
            Tensor actH = new Tensor(projH.Shape, DType.F16); backend.Gelu(actH, projH); projH.Dispose();
            Tensor outH = new Tensor(new TensorShape(s, _dim), DType.F16); backend.Linear(outH, actH, _ffOutW!, _ffOutB);
            actH.Dispose();
            Tensor outF = new Tensor(new TensorShape(s, _dim), DType.F32); backend.CastToF32(outF, outH);
            outH.Dispose();
            return outF;
        }
        Tensor proj = new Tensor(new TensorShape(s, inner), x.DType); backend.Linear(proj, x, _ffProjW!, _ffProjB);
        Tensor act = new Tensor(proj.Shape, x.DType); backend.Gelu(act, proj); proj.Dispose();
        Tensor outT = new Tensor(new TensorShape(s, _dim), x.DType); backend.Linear(outT, act, _ffOutW!, _ffOutB);
        act.Dispose();
        return outT;
    }

    /// <summary>Adds <c>scale_shift_table</c> to the timestep_proj <c>[G, 6, dim]</c>; returns 6 modulation tensors of <c>[G, dim]</c>.
    /// <para>G=1 (scalar-timestep T2V/I2V, the hot path) builds the 6 tensors <b>on-device</b> via one <see cref="IBackend.Add"/>
    /// over the table + <see cref="IBackend.SliceRows"/>, so the results are GPU-resident activations. This is a load-bearing perf
    /// fix: the old host-pointer loop produced fresh CPU tensors whose first <c>CopyToDevice</c> (in every downstream AddScalar /
    /// AffineBroadcast / GatedResidual) is a cache MISS, and the miss path does a full stream <c>SyncStream</c> before its H2D —
    /// so each block drained the whole async attention/FFN pipeline 3× (measured: ~85 s of a ~90 s Wan-1.3B gen sat in that
    /// GatedResidual sync). Keeping modulation on-device makes every downstream upload a cache HIT → no per-block barriers.
    /// The multi-group TI2V path (per-frame timesteps) keeps the CPU reference.</para></summary>
    private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) Modulation(IBackend backend, Tensor temb, bool scalePlusOne)
    {
        int g = (int)temb.Shape[0];
        if (g == 1)
        {
            // out[m] = temb[0,m,:] + scale_shift_table[m,:], each [1, dim], GPU-resident; with scalePlusOne the two scale rows are (1 + scale).
            using Tensor packed = new Tensor(new TensorShape(6, _dim), DType.F32);
            backend.Add(packed, temb, _scaleShift!);
            using Tensor? packedP1 = scalePlusOne ? new Tensor(packed.Shape, DType.F32) : null;
            if (packedP1 is not null) backend.AddScalar(packedP1, packed, 1.0f);
            Tensor scales = packedP1 ?? packed;
            return (Row(backend, packed, 0), Row(backend, scales, 1), Row(backend, packed, 2),
                Row(backend, packed, 3), Row(backend, scales, 4), Row(backend, packed, 5));
        }
        // Multi-group (per-frame timesteps: S2V/TI2V/Animate) — GPU-resident since 2026-07-09. The old
        // host loop here was the S2V 15x-vs-T2V wall: fresh CPU tensors -> downstream CopyToDevice cache
        // MISS -> full stream drain, ~4 big bounces per block per forward (profiled 324 H2D_MISS_BIG/step).
        // Permute [G,6,dim] -> [6,G,dim] so each modulation's G rows are contiguous, then slice + broadcast-add.
        Tensor ones6 = Ones();
        Tensor perm = new Tensor(new TensorShape(6 * g, _dim), DType.F32);
        backend.Permute0213(perm, temb, g, 6, _dim);        // [1,G,6,dim] -> [1,6,G,dim]
        Tensor[] outsG = new Tensor[6];
        for (int m = 0; m < 6; m++)
        {
            Tensor tbM = new Tensor(new TensorShape(g, _dim), DType.F32);
            backend.SliceRows(tbM, perm, m * g);            // temb[:, m, :]
            Tensor ssM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(ssM, _scaleShift!, m);        // scale_shift_table row m
            Tensor o = new Tensor(new TensorShape(g, _dim), DType.F32);
            backend.AffineBroadcastLastDim(o, tbM, ones6, ssM);   // o = tbM*1 + ssM (row-broadcast)
            tbM.Dispose(); ssM.Dispose();
            outsG[m] = o;
        }
        perm.Dispose();
        return (outsG[0], outsG[1], outsG[2], outsG[3], outsG[4], outsG[5]);
    }

    private Tensor Row(IBackend backend, Tensor table, int row)
    {
        Tensor o = new Tensor(new TensorShape(1, _dim), DType.F32);
        backend.SliceRows(o, table, row);
        return o;
    }

    /// <summary>Expands per-group rows <c>[G, dim]</c> to per-token rows <c>[G*tokensPerGroup, dim]</c> by row
    /// repetition — GPU-resident via <see cref="IBackend.UpsampleNearest2D"/> over an NCHW view.</summary>
    private Tensor ExpandGroups(IBackend backend, Tensor grouped, int groups, int tokensPerGroup)
    {
        Tensor src = new Tensor(new TensorShape(1, 1, groups, _dim), DType.F32);
        backend.SliceRows(src, grouped, 0);
        Tensor up = new Tensor(new TensorShape(1, 1, (long)groups * tokensPerGroup, _dim), DType.F32);
        backend.UpsampleNearest2D(up, src, tokensPerGroup, 1);
        src.Dispose();
        Tensor outT = new Tensor(new TensorShape((long)groups * tokensPerGroup, _dim), DType.F32);
        backend.SliceRows(outT, up, 0);
        up.Dispose();
        return outT;
    }

    /// <summary>FP32 LayerNorm over the last dim with optional affine, written in <paramref name="output"/>. GPU-resident (backend
    /// ops) — the per-op host-pointer loop it replaced dominated full-res runtime (14040×3072 per call, forcing D2H/H2D each block).
    /// An F16 output with weight and bias is one <see cref="IBackend.LayerNorm"/> pass; otherwise no-affine →
    /// <see cref="IBackend.LayerNormNoAffine"/>, affine → that plus <see cref="IBackend.AffineBroadcastLastDim"/>.</summary>
    private Tensor LayerNorm(IBackend backend, Tensor x, Tensor? weight, Tensor? bias, int s, DType output)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), output);
        if (output != x.DType && weight is not null && bias is not null)
        {
            backend.LayerNorm(o, x, weight, bias, _eps);
            return o;
        }
        // The affine step must NOT be in-place: re-caching the same tensor orphans its old device buffer (FreeDevice
        // skips cached pointers), leaking it — so normalize into a scratch tensor, then affine into the output.
        if (weight is null)
        {
            backend.LayerNormNoAffine(o, x, _eps);
            return o;
        }
        using Tensor normed = new Tensor(new TensorShape(s, _dim), x.DType);
        backend.LayerNormNoAffine(normed, x, _eps);
        if (output == x.DType)
        {
            backend.AffineBroadcastLastDim(o, normed, weight, bias);
            return o;
        }
        using Tensor affine = new Tensor(new TensorShape(s, _dim), x.DType);
        backend.AffineBroadcastLastDim(affine, normed, weight, bias);
        backend.CastToF16(o, affine);
        return o;
    }

    /// <summary>adaLN of the F32 stream into an F16 branch, <c>LN(x)·(1 + scale) + shift</c>, in one pass that rounds once.</summary>
    private Tensor NormModulateF16(IBackend backend, Tensor x, Tensor scale, Tensor shift, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F16);
        backend.LayerNormModulate(o, x, scale, shift, _eps);
        return o;
    }

    /// <summary>AdaLN modulate: <c>out = x·(1+scale) + shift</c>, scale/shift broadcast per group over the last dim.
    /// Consumes <paramref name="x"/> (disposes it) and returns the result. G=1 (scalar timestep) is GPU-resident via
    /// <see cref="IBackend.AffineBroadcastLastDim"/> (pre-adding 1 to scale on-GPU unless <paramref name="scalePlusOne"/>
    /// says the caller did) into a FRESH tensor — never in-place, which would orphan/leak x's cached device buffer. The
    /// multi-group TI2V path keeps the CPU reference (mutates x in place and returns it).</summary>
    private Tensor ApplyShiftScale(IBackend backend, Tensor x, Tensor scale, Tensor shift, int s, int tokensPerGroup, bool scalePlusOne)
    {
        if (tokensPerGroup == s)   // G == 1: scale/shift are [1, dim] broadcast over all S tokens
        {
            using Tensor? scaleP1 = scalePlusOne ? null : new Tensor(scale.Shape, DType.F32);
            if (scaleP1 is not null) backend.AddScalar(scaleP1, scale, 1.0f);           // (1 + scale)
            Tensor o = new Tensor(new TensorShape(s, _dim), x.DType);
            backend.AffineBroadcastLastDim(o, x, scaleP1 ?? scale, shift);   // o = x·(1+scale) + shift  (non-in-place)
            x.Dispose();
            return o;
        }
        int groupsSS = (int)scale.Shape[0];
        if ((long)groupsSS * tokensPerGroup == s)
        {
            // GPU multi-group path (2026-07-09): expand scale/shift to token rows, then elementwise FMA.
            using Tensor scaleExp = ExpandGroups(backend, scale, groupsSS, tokensPerGroup);
            using Tensor shiftExp = ExpandGroups(backend, shift, groupsSS, tokensPerGroup);
            using Tensor scaleP1 = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.AddScalar(scaleP1, scaleExp, 1.0f);
            using Tensor prod = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Mul(prod, x, scaleP1);
            Tensor oG = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Add(oG, prod, shiftExp);
            x.Dispose();
            return oG;
        }
        float* xp = (float*)x.DataPointer; float* sc = (float*)scale.DataPointer; float* sh = (float*)shift.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) xp[i * _dim + d] = xp[i * _dim + d] * (1f + sc[g + d]) + sh[g + d];
        }
        return x;
    }

    /// <summary>Gated residual: <c>out = residual + gate·value</c>, gate broadcast per group over the last dim.
    /// G=1 is GPU-resident via <see cref="IBackend.GatedResidualLastDim"/>; multi-group TI2V keeps the CPU reference.</summary>
    private Tensor GatedAdd(IBackend backend, Tensor residual, Tensor value, Tensor gate, int s, int tokensPerGroup)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), residual.DType);
        if (tokensPerGroup == s)   // G == 1: gate is [1, dim]
        {
            backend.GatedResidualLastDim(o, residual, value, gate);
            return o;
        }
        int groupsGA = (int)gate.Shape[0];
        if ((long)groupsGA * tokensPerGroup == s)
        {
            // GPU multi-group path (2026-07-09): expand the gate to token rows, then elementwise ops.
            using Tensor gateExp = ExpandGroups(backend, gate, groupsGA, tokensPerGroup);
            using Tensor prod = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Mul(prod, value, gateExp);
            backend.Add(o, residual, prod);
            return o;
        }
        float* rp = (float*)residual.DataPointer; float* vp = (float*)value.DataPointer; float* gp = (float*)gate.DataPointer; float* op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) op[i * _dim + d] = rp[i * _dim + d] + gp[g + d] * vp[i * _dim + d];
        }
        return o;
    }

    // Cached [1, dim] ones row, the gate that runs a plain add as GatedResidualLastDim: out = a + 1·b.
    // Built once on first use; fill-before-publish + CAS because two
    // branch/rank threads (CFG-parallel, context-parallel) share the block and race the first touch.
    private Tensor? _ones;
    private Tensor Ones()
    {
        if (_ones is null)
        {
            Tensor ones = new Tensor(new TensorShape(1, _dim), DType.F32);
            float* p = (float*)ones.DataPointer;
            for (int i = 0; i < _dim; i++) p[i] = 1f;
            if (Interlocked.CompareExchange(ref _ones, ones, null) is not null) ones.Dispose();
        }
        return _ones;
    }

    private Tensor AddRows(IBackend backend, Tensor a, Tensor b, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), a.DType);
        backend.GatedResidualLastDim(o, a, b, Ones());   // a + 1·b  (GPU)
        return o;
    }

    // [s, dim]=[s, heads, headDim] → [1, heads, s, headDim], GPU-resident via Permute0213 (explicit dims, reads the
    // device buffer directly — correct now that RoPE is GPU so x never leaves the device).
    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    // [1, heads, s, headDim] → [s, dim]=[s, heads, headDim], GPU-resident via Permute0213 (inverse of ToBhsd).
    private Tensor FromBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Permute0213(o, x, _heads, s, _headDim);
        return o;
    }
}
