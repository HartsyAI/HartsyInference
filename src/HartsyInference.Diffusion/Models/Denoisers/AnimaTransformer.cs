using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Anima / Cosmos-Predict2-2B-Text2Image DiT (image-only mode). Every tensor name + shape below is measured
/// from the actual <c>anima-preview3-base.safetensors</c> ComfyUI single-file checkpoint. All weights live under a
/// flat <c>net.</c> prefix in the file; the <see cref="HartsyInference.ModelAssets.CheckpointConverters.AnimaCheckpointConverter"/>
/// strips that prefix before handing us the dict, so the loader expects keys like <c>x_embedder.proj.1.weight</c>,
/// <c>blocks.0.self_attn.q_proj.weight</c>, <c>final_layer.linear.weight</c>.
///
/// Forward summary:
/// <list type="number">
///   <item>Concat a constant-zero padding-mask channel → <c>[B, 17, H, W]</c>.</item>
///   <item>Patch embed: gather <c>2×2</c> patches, flatten to 68-dim features per token, then
///        <c>x_embedder.proj.1</c> Linear[2048, 68] → <c>[B, S, 2048]</c>. No bias.</item>
///   <item>Time embedding: sin/cos timesteps → RMSNorm (<c>t_embedding_norm.weight [2048]</c>) → <c>embedded_timestep [B, 2048]</c>.
///        Same sin/cos → <c>t_embedder.1.linear_1 [2048,2048]</c> → SiLU → <c>t_embedder.1.linear_2 [6144,2048]</c> → <c>temb [B, 6144]</c>.</item>
///   <item>28 stacked <see cref="AnimaBlock"/>s, each with three AdaLN-LoRA modulators and self+cross+mlp sub-blocks.
///        Cross-attention K/V come from the <see cref="AnimaLlmAdapter"/> output (1024-dim).</item>
///   <item>Final layer (<c>final_layer.adaln_modulation</c>): AdaLN-LoRA with 2-chunk output (shift, scale only, no gate).
///        Linear[256, 2048] → Linear[4096, 256] → + temb[:4096] → chunk(2) → (shift, scale). Apply <c>norm(x) * (1+scale) + shift</c>.</item>
///   <item><c>final_layer.linear</c> projects each token to <c>p_h × p_w × out_channels = 64</c>, then unpatchify back to
///        <c>[B, out_channels, H, W]</c>.</item>
/// </list>
///
/// All DiT-trunk weights in the Anima checkpoint are bias-less.</summary>
public sealed unsafe class AnimaTransformer : IDisposable
{
    private readonly AnimaConfig _config;
    private readonly AnimaBlock[] _blocks;
    private readonly AnimaRope _rope;
    private int _disposed;

    // ── x_embedder.proj.1.weight [hidden, in_total] where in_total = (in_ch + concat_mask?) × p_h × p_w. ──
    private Tensor? _patchEmbedWeight;  // No bias in Anima checkpoint.

    // ── t_embedder.1.linear_1.weight [hidden, hidden], linear_2.weight [condition_dim, hidden]. No biases. ──
    private Tensor? _timeLinear1Weight;
    private Tensor? _timeLinear2Weight;

    // ── t_embedding_norm.weight [hidden] (RMSNorm scale applied to the raw sin/cos vector). ──
    private Tensor? _timeEmbeddingNormWeight;

    // ── final_layer.adaln_modulation: .1.weight [rank, hidden], .2.weight [2*hidden, rank]. No biases. ──
    private Tensor? _finalAdaL1, _finalAdaL2;

    // ── final_layer.linear.weight [p_t*p_h*p_w*out_ch, hidden]. No bias. ──
    private Tensor? _finalLinearWeight;

    public AnimaTransformer(AnimaConfig config)
    {
        if (config.PatchSize.T != 1)
            throw new ArgumentException(
                $"AnimaTransformer is t2i-only; PatchSize.T must be 1 (got {config.PatchSize.T}).",
                nameof(config));
        if (config.PatchSize.H <= 0 || config.PatchSize.W <= 0)
            throw new ArgumentException(
                $"AnimaTransformer patch dimensions must be positive (got {config.PatchSize.H}x{config.PatchSize.W}).",
                nameof(config));
        if (config.PatchSize.H != config.PatchSize.W)
            throw new ArgumentException(
                $"AnimaTransformer requires the checkpoint's square image patch (got {config.PatchSize.H}x{config.PatchSize.W}).",
                nameof(config));
        _config = config;
        _rope = new AnimaRope(config.HeadDim, config.RopeTheta, config.RopeScale);
        _blocks = new AnimaBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new AnimaBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim, config.FfnHiddenSize,
                config.AdaLnLoraDim, config.RmsNormEps, config.QkNormEps);
        }
    }

    public AnimaConfig Config => _config;

    /// <summary>Loads DiT-trunk weights from the converter-stripped dict (no <c>net.</c> prefix).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Patch embed: wrapped in nn.Sequential([?, Linear]); the linear is at index .1.
        _patchEmbedWeight = weights["x_embedder.proj.1.weight"];

        // Time embedder: wrapped in nn.Sequential at .1. Linear_1[hidden, hidden] → SiLU → Linear_2[condition_dim, hidden].
        _timeLinear1Weight = weights["t_embedder.1.linear_1.weight"];
        _timeLinear2Weight = weights["t_embedder.1.linear_2.weight"];

        // RMSNorm of the raw sin/cos vector → embedded_timestep.
        _timeEmbeddingNormWeight = TensorCasts.LoadF32(weights, "t_embedding_norm.weight");

        // Final-layer AdaLN-LoRA (2-chunk: shift, scale).
        _finalAdaL1 = weights["final_layer.adaln_modulation.1.weight"];
        _finalAdaL2 = weights["final_layer.adaln_modulation.2.weight"];

        // Final patch-projection linear.
        _finalLinearWeight = weights["final_layer.linear.weight"];

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbedWeight is not null) yield return _patchEmbedWeight;
        if (_timeLinear1Weight is not null) yield return _timeLinear1Weight;
        if (_timeLinear2Weight is not null) yield return _timeLinear2Weight;
        if (_timeEmbeddingNormWeight is not null) yield return _timeEmbeddingNormWeight;
        if (_finalAdaL1 is not null) yield return _finalAdaL1;
        if (_finalAdaL2 is not null) yield return _finalAdaL2;
        if (_finalLinearWeight is not null) yield return _finalLinearWeight;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Releases tensors created lazily by this transformer that are not checkpoint weights; call only after the backend stream is synchronized, since the next forward recreates and preloads the current grid's tables.</summary>
    public void ReleaseDeviceCache(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _rope.ReleaseGpuTables(backend);
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">VAE latent <c>[B, in_channels=16, H, W]</c>.</param>
    /// <param name="timestep">Scalar timestep.</param>
    /// <param name="textEmbeds">Refined text features from <see cref="AnimaLlmAdapter"/>, <c>[B, T, 1024]</c>.</param>
    /// <param name="textAttentionMask">Optional padded-text mask <c>[B, T]</c> (1=valid, 0=padded). When non-null,
    /// converted to an additive cross-attn mask.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int[]? textAttentionMask = null)
    {
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4D [B, C, H, W], got {latent.Shape}.", nameof(latent));
        if (textEmbeds.Shape.Rank != 3)
            throw new ArgumentException($"textEmbeds must be 3D [B, T, dim], got {textEmbeds.Shape}.", nameof(textEmbeds));
        if (latent.DType != DType.F32 || textEmbeds.DType != DType.F32)
            throw new NotSupportedException(
                $"AnimaTransformer requires F32 latent/text activations; got latent={latent.DType}, text={textEmbeds.DType}.");
        if (latent.Shape[0] <= 0 || latent.Shape[1] <= 0 || latent.Shape[2] <= 0 || latent.Shape[3] <= 0
            || textEmbeds.Shape[0] <= 0 || textEmbeds.Shape[1] <= 0 || textEmbeds.Shape[2] <= 0)
            throw new ArgumentException($"AnimaTransformer dimensions must be positive; got latent={latent.Shape}, text={textEmbeds.Shape}.");
        if (latent.Shape[0] != textEmbeds.Shape[0])
            throw new ArgumentException(
                $"Latent/text batch dimensions must match; got latent={latent.Shape[0]}, text={textEmbeds.Shape[0]}.",
                nameof(textEmbeds));
        if (latent.Shape[1] != _config.InChannels)
            throw new ArgumentException(
                $"Latent channel dimension must equal configured InChannels={_config.InChannels}; got {latent.Shape[1]}.",
                nameof(latent));
        if (textEmbeds.Shape[2] != _config.LlmAdapter.HiddenSize)
            throw new ArgumentException(
                $"Text feature width must equal LLM-adapter hidden size {_config.LlmAdapter.HiddenSize}; got {textEmbeds.Shape[2]}.",
                nameof(textEmbeds));
        if (!float.IsFinite(timestep))
            throw new ArgumentOutOfRangeException(nameof(timestep), "Anima timestep must be finite.");

        int batch = (int)latent.Shape[0];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int hidden = _config.HiddenSize;
        int patchH = _config.PatchSize.H;
        int patchW = _config.PatchSize.W;
        int gridH = latentH / patchH;
        int gridW = latentW / patchW;
        int seqLen = gridH * gridW;
        int textSeq = (int)textEmbeds.Shape[1];

        if (latentH % patchH != 0 || latentW % patchW != 0)
            throw new ArgumentException(
                $"Latent {latentH}x{latentW} not divisible by patch {patchH}x{patchW}.", nameof(latent));

        System.Diagnostics.Stopwatch phaseSw = System.Diagnostics.Stopwatch.StartNew();
        // ── 1. Concat padding-mask channel → [B, 17, H, W] ──
        Tensor latentExpanded = _config.ConcatPaddingMask ? AppendPaddingMaskChannel(backend, latent) : latent;
        AnimaDebugDump.Dump("latent_with_mask", latentExpanded);

        int totalChannels = (int)latentExpanded.Shape[1];

        // ── 2. Patch embed → [B, S, hidden] ──
        Tensor imgTokens;
        try
        {
            imgTokens = PatchEmbed(backend, latentExpanded, batch, totalChannels, gridH, gridW);
        }
        finally
        {
            if (!ReferenceEquals(latentExpanded, latent))
                DisposeBestEffort(latentExpanded, "expanded latent");
        }
        AnimaDebugDump.Dump("patch_embed", imgTokens);

        long tPatch = phaseSw.ElapsedMilliseconds;
        // ── 3. RoPE table for image self-attn ──
        // The pair is immutable for a grid and borrowed from _rope's per-backend cache. Explicit preload keeps
        // all 28 blocks and later denoising steps on the same device addresses.
        (Tensor ropeCos, Tensor ropeSin) = _rope.GetOrCreateTables(
            backend, tFrames: 1, hPatched: gridH, wPatched: gridW);
        long tRope = phaseSw.ElapsedMilliseconds;
        AnimaDebugDump.Dump("rope_cos", ropeCos);
        AnimaDebugDump.Dump("rope_sin", ropeSin);

        // ── 4. Time embedding (temb [B, 6144], embedded_timestep [B, 2048]) ──
        (Tensor temb, Tensor embeddedTimestep) = TimeEmbed(backend, timestep, batch);
        AnimaDebugDump.Dump("temb", temb);
        AnimaDebugDump.Dump("embedded_timestep", embeddedTimestep);

        // ── 5. Build cross-attn mask if provided ──
        Tensor? crossMask = textAttentionMask is not null ? BuildAttentionMask(batch, textSeq, textAttentionMask)
            : null;

        // ── 6. Layer loop ──
        Tensor x = imgTokens;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, textEmbeds, embeddedTimestep, temb, ropeCos, ropeSin, _rope, crossMask);
            x.Dispose();
            x = next;
            AnimaDebugDump.Dump($"block_{i}", x);
        }

        long tBlocks = phaseSw.ElapsedMilliseconds;
        crossMask?.Dispose();

        // ── 7. Final AdaLN-LoRA (2-chunk) + linear ──
        Tensor projected = ApplyFinalLayer(backend, x, embeddedTimestep, temb, batch, seqLen);
        x.Dispose();
        temb.Dispose();
        embeddedTimestep.Dispose();
        AnimaDebugDump.Dump("final_proj", projected);

        // ── 8. Unpatchify ──
        long tFinal = phaseSw.ElapsedMilliseconds;
        Tensor velocity;
        try
        {
            velocity = Unpatchify(backend, projected, batch, gridH, gridW, patchH, patchW, _config.OutChannels);
        }
        finally
        {
            DisposeBestEffort(projected, "projected patch tokens");
        }
        AnimaDebugDump.DumpOutput(velocity);
        HartsyInference.Core.Logging.Logs.Verbose($"[anima-phase] patch={tPatch}ms rope={tRope - tPatch}ms blocks={tBlocks - tRope}ms final={tFinal - tBlocks}ms unpatch={phaseSw.ElapsedMilliseconds - tFinal}ms");

        return velocity;
    }

    /// <summary>Concatenates a zero-valued padding-mask channel to <c>[B, C, H, W]</c>, returning
    /// <c>[B, C+1, H, W]</c>. Upstream Cosmos (per <c>transformer_cosmos.py</c> line 707-712)
    /// resizes the caller-supplied <c>padding_mask</c> to the latent's spatial dim and concatenates
    /// it. For unpadded t2i (the only mode we support), the padding mask is all zeros — there's no
    /// padding to ignore. Using a constant <b>0</b> here, not 1, matches what the model was trained
    /// to expect on the extra channel.</summary>
    private static Tensor AppendPaddingMaskChannel(IBackend backend, Tensor latent)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        Tensor mask = new(new TensorShape(batch, 1, height, width), DType.F32);
        Tensor output = new(new TensorShape(batch, channels + 1, height, width), DType.F32);
        bool success = false;
        try
        {
            backend.Fill(mask, 0f);
            backend.Concat(output, [latent, mask], dim: 1);
            success = true;
            return output;
        }
        finally
        {
            DisposeBestEffort(mask, "padding mask");
            if (!success) DisposeBestEffort(output, "expanded latent after failure");
        }
    }

    /// <summary>Patch embed via a single linear projection over the flattened p_t·p_h·p_w-element patch.</summary>
    private Tensor PatchEmbed(IBackend backend, Tensor latent, int batch, int channels, int gridH, int gridW)
    {
        int patchH = _config.PatchSize.H;
        int patchW = _config.PatchSize.W;
        int patchVol = patchH * patchW;
        int patchFeat = channels * patchVol;

        TensorShape gatherShape = new TensorShape(batch, gridH * gridW, patchFeat);
        Tensor gathered = new Tensor(gatherShape, DType.F32);
        Tensor? output = null;
        bool success = false;
        try
        {
            // Cosmos patch embedding flattens each patch as (channel, ph, pw): channel OUTERMOST.
            // PatchifyTokens(false) is that exact byte order and keeps the carried latent on the device.
            backend.PatchifyTokens(gathered, latent, patchH, innerChannelFastest: false);

            TensorShape outShape = new TensorShape(batch, gridH * gridW, _config.HiddenSize);
            output = new Tensor(outShape, DType.F32);
            backend.Linear(output, gathered, _patchEmbedWeight!, null);
            success = true;
            return output;
        }
        finally
        {
            DisposeBestEffort(gathered, "patch-gather staging tensor");
            if (!success) DisposeBestEffort(output, "patch-embedding output after failure");
        }
    }

    /// <summary>Time embedding: sin/cos timesteps → MLP produces <c>temb [B, condition_dim=6144]</c>; the raw sin/cos
    /// is RMSNorm'd to produce <c>embedded_timestep [B, hidden=2048]</c>.</summary>
    private (Tensor Temb, Tensor EmbeddedTimestep) TimeEmbed(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;
        int condDim = _config.ConditionDim;

        // Sin/cos timesteps at `hidden` width (matches t_embedder.1.linear_1's input dim).
        TensorShape sinShape = new TensorShape(batch, hidden);
        Tensor sinEmb = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmb, timestep, batch, hidden, 10000f);
        AnimaDebugDump.Dump("time_sin_proj", sinEmb);

        // MLP: Linear[hidden, hidden] → SiLU → Linear[condDim, hidden] → temb.
        Tensor mlp1 = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(mlp1, sinEmb, _timeLinear1Weight!, null);

        Tensor mlp1Act = new Tensor(mlp1.Shape, DType.F32);
        backend.Silu(mlp1Act, mlp1);
        mlp1.Dispose();

        Tensor temb = new Tensor(new TensorShape(batch, condDim), DType.F32);
        backend.Linear(temb, mlp1Act, _timeLinear2Weight!, null);
        mlp1Act.Dispose();

        // embedded_timestep: RMSNorm of the raw sin/cos vector (scale = t_embedding_norm.weight).
        Tensor embeddedTimestep = new Tensor(sinShape, DType.F32);
        backend.RmsNorm(embeddedTimestep, sinEmb, _timeEmbeddingNormWeight!, _config.RmsNormEps);
        sinEmb.Dispose();

        return (temb, embeddedTimestep);
    }

    /// <summary>Builds an additive cross-attention mask <c>[B, 1, 1, T]</c> (0 valid, large-negative masked).</summary>
    private static Tensor BuildAttentionMask(int batch, int textSeq, int[] textAttentionMask)
    {
        if (textAttentionMask.Length != batch * textSeq && textAttentionMask.Length != textSeq)
            throw new ArgumentException(
                $"textAttentionMask length {textAttentionMask.Length} must be batch*textSeq ({batch * textSeq}) or just textSeq ({textSeq}).",
                nameof(textAttentionMask));

        TensorShape shape = new TensorShape(batch, 1, 1, textSeq);
        Tensor mask = new Tensor(shape, DType.F32);
        const float negInf = -1e30f;
        float* p = (float*)mask.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int srcBase = textAttentionMask.Length == textSeq ? 0 : b * textSeq;
            long off = (long)b * textSeq;
            for (int t = 0; t < textSeq; t++)
                p[off + t] = textAttentionMask[srcBase + t] != 0 ? 0f : negInf;
        }
        return mask;
    }

    /// <summary>Final-layer AdaLN-LoRA + linear. The AdaLN here outputs 2 chunks (shift, scale only — no gate):
    /// <c>Linear[rank, hidden] → Linear[2*hidden, rank] → + temb[:2*hidden] → chunk(2)</c>. Then
    /// <c>norm(x) * (1+scale) + shift</c>, then <c>final_layer.linear</c> projects per-token to <c>p_h*p_w*out_ch</c>.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor x, Tensor embeddedTimestep, Tensor temb, int batch, int seqLen)
    {
        int hidden = _config.HiddenSize;
        int rank = _config.AdaLnLoraDim;
        int expandWidth = 2 * hidden;  // shift + scale only.

        // silu(embedded_timestep) [B, hidden]
        Tensor act = new Tensor(embeddedTimestep.Shape, DType.F32);
        backend.Silu(act, embeddedTimestep);

        // bottleneck [B, rank]
        Tensor rankProj = new Tensor(new TensorShape(batch, rank), DType.F32);
        backend.Linear(rankProj, act, _finalAdaL1!, null);
        act.Dispose();

        // expand [B, 2*hidden]
        Tensor expanded = new Tensor(new TensorShape(batch, expandWidth), DType.F32);
        backend.Linear(expanded, rankProj, _finalAdaL2!, null);
        rankProj.Dispose();

        // + temb[:2*hidden]
        Tensor modulation = AnimaBlock.AddTembSlice(backend, expanded, temb, batch, expandWidth);
        expanded.Dispose();

        // chunk(2) → shift, scale
        Tensor shift = new Tensor(new TensorShape(batch, hidden), DType.F32);
        Tensor scale = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.SliceLastDim(shift, modulation, 0);
        backend.SliceLastDim(scale, modulation, hidden);
        modulation.Dispose();

        // LayerNorm-no-affine + modulate, fused on-device.
        TensorShape seqShape = new TensorShape(batch, seqLen, hidden);
        Tensor modulated = new Tensor(seqShape, DType.F32);
        backend.LayerNormModulate(modulated, x, scale, shift, _config.RmsNormEps);
        shift.Dispose();
        scale.Dispose();

        // final_layer.linear: hidden → p_h * p_w * out_channels (p_t = 1). Shape [out_features=64, in_features=2048].
        int outDim = _config.PatchSize.H * _config.PatchSize.W * _config.OutChannels;
        Tensor output = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, modulated, _finalLinearWeight!, null);
        modulated.Dispose();
        return output;
    }

    /// <summary>Reshapes <c>[B, S, p_h*p_w*C]</c> back to <c>[B, C, H, W]</c>. Mirrors the upstream
    /// Cosmos <c>permute(0, 7, 1, 6, 2, 4, 3, 5)</c> ordering for the proj_out output (per
    /// <c>diffusers/models/transformers/transformer_cosmos.py</c> line 800-807). Crucial detail:
    /// PATCHIFY uses feature order <c>(C, p_t, p_h, p_w)</c> with channel OUTERMOST, but
    /// UNPATCHIFY uses <c>(p_h, p_w, p_t, C)</c> with channel INNERMOST. The two orderings are
    /// deliberately asymmetric — the upstream comment in <c>transformer_cosmos.py</c> calls this
    /// out as "NOT the inverse operation of what happens when patching as usually expected". So
    /// flat feature index in each token's <c>p_h*p_w*p_t*C</c>-element vector is:
    /// <code>feat_idx = ph * (p_w * p_t * C) + pw * (p_t * C) + pt * C + c</code>.
    /// For image mode (p_t = 1) this collapses to <c>ph * p_w * C + pw * C + c</c>.</summary>
    private static Tensor Unpatchify(IBackend backend, Tensor packed, int batch, int gridH, int gridW, int patchH, int patchW, int outChannels)
    {
        int height = gridH * patchH;
        int width = gridW * patchW;
        TensorShape shape = new TensorShape(batch, outChannels, height, width);
        Tensor output = new Tensor(shape, DType.F32);
        bool success = false;
        try
        {
            // Cosmos final projection is intentionally NOT the inverse of patch embedding: its token
            // feature order is (ph, pw, channel), i.e. channel INNERMOST.
            backend.UnpatchifyTokens(
                output, packed, outChannels, gridH, gridW, patchH, innerChannelFastest: true);
            success = true;
            return output;
        }
        finally
        {
            if (!success) DisposeBestEffort(output, "unpatchified output after failure");
        }
    }

    private static void DisposeBestEffort(Tensor? tensor, string description)
    {
        if (tensor is null) return;
        try { tensor.Dispose(); }
        catch (Exception error)
        {
            HartsyInference.Core.Logging.Logs.Error($"Failed to release Anima transformer {description}.", error);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchEmbedWeight = null;
            _timeLinear1Weight = null;
            _timeLinear2Weight = null;
            _timeEmbeddingNormWeight = null;
            _finalAdaL1 = null;
            _finalAdaL2 = null;
            _finalLinearWeight = null;
        }
        GC.SuppressFinalize(this);
    }
}
