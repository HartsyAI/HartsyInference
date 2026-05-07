using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Anima / Cosmos-Predict2-2B-Text2Image diffusion transformer (image-only mode). Implements the diffusers
/// <c>CosmosTransformer3DModel</c> with the temporal axis pinned to <c>T = 1</c>. Drops every video / world-model code
/// path: no FPS-conditioned RoPE, no multi-frame patchify, no <c>condition_mask</c> autoregressive conditioning, no
/// ControlNet residuals, no image-context (Predict2.5) hybrid attention. Supports only standard t2i forward.
///
/// Forward summary (per <c>diffusers/models/transformers/transformer_cosmos.py</c>):
/// <list type="number">
///   <item>Optional concat of a constant-1 padding-mask channel (<c>concat_padding_mask=True</c>) → <c>[B, C+1, 1, H, W]</c>.</item>
///   <item>RoPE table: 3-axis, but <c>T = 1</c> collapses temporal sub-bands to position 0.</item>
///   <item>Patch embed: <c>Linear((C+1) * p_t * p_h * p_w, hidden)</c> over the flattened <c>p_t·p_h·p_w</c>-element patch.</item>
///   <item>Optional learnable positional embedding (<c>extra_pos_embed_type="learnable"</c>) — sum of three axis-specific
///        learned tables, then L2-normalized over the last dim — added to the patch sequence.</item>
///   <item>Time embed: sin/cos timesteps → <c>CosmosTimestepEmbedding(Linear→SiLU→Linear)</c> = <c>temb [B, condition_dim]</c>;
///        the pre-MLP sin/cos vector is RMSNorm'd to <c>embedded_timestep [B, hidden]</c>.</item>
///   <item>Optional <c>crossattn_proj</c>: Linear maps text encoder hiddens to <c>hidden</c>.</item>
///   <item>28 stacked <see cref="AnimaBlock"/>s.</item>
///   <item>Final <c>CosmosAdaLayerNorm</c> (Linear→+temb[:2*hidden]→chunk(2,-1)→x*(1+scale)+shift) + <c>proj_out</c> Linear →
///        unpatchify back to <c>[B, C_out, H, W]</c>.</item>
/// </list>
///
/// **Image-only invariants (asserted at runtime):** <c>p_t = 1</c>, latent rank is 4 (<c>[B, C, H, W]</c>), <c>fps</c>
/// is unused.</summary>
public sealed unsafe class AnimaTransformer : IDisposable
{
    private readonly AnimaConfig _config;
    private readonly AnimaBlock[] _blocks;
    private readonly AnimaRope _rope;
    private int _disposed;

    // patch_embed.proj: Linear((in_channels + concat_mask?) * p_t * p_h * p_w, hidden) with bias.
    private Tensor? _patchEmbedWeight, _patchEmbedBias;

    // crossattn_proj (optional): Linear(text_embed_dim, hidden).
    private Tensor? _crossAttnProjWeight, _crossAttnProjBias;

    // time_embed: Timesteps(sin/cos, embedding_dim, flip_sin_to_cos=True) — no params; followed by:
    //   t_embedder.linear_1 [hidden, hidden]
    //   t_embedder.linear_2 [condition_dim, hidden]
    //   norm: RMSNorm(embedding_dim)
    private Tensor? _timeLinear1Weight, _timeLinear1Bias;
    private Tensor? _timeLinear2Weight, _timeLinear2Bias;
    private Tensor? _timeNormWeight; // RMSNorm scale, [hidden]

    // learnable_pos_embed (optional, [max_t, dim] / [max_h, dim] / [max_w, dim]).
    private Tensor? _posEmbT, _posEmbH, _posEmbW;

    // norm_out: Linear(condition_dim, 2*hidden) + bias. Followed by + temb[:2*hidden], chunk(2,-1) → scale, shift.
    private Tensor? _normOutLinear1Weight, _normOutLinear1Bias;
    private Tensor? _normOutLinear2Weight, _normOutLinear2Bias;

    // proj_out: Linear(hidden, p_t * p_h * p_w * out_channels).
    private Tensor? _projOutWeight, _projOutBias;

    public AnimaTransformer(AnimaConfig config)
    {
        if (config.PatchSize.T != 1)
            throw new ArgumentException(
                $"AnimaTransformer is t2i-only; PatchSize.T must be 1 (got {config.PatchSize.T}). Video paths are not implemented.",
                nameof(config));
        _config = config;
        _rope = new AnimaRope(config.HeadDim, config.RopeTheta, config.RopeScale);
        _blocks = new AnimaBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new AnimaBlock(config.HiddenSize, config.NumHeads, config.FfnHiddenSize, config.RmsNormEps, config.QkNormEps);
    }

    /// <summary>Convenience accessor for the config bound at construction.</summary>
    public AnimaConfig Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict (post-converter). Tolerates missing optional keys for
    /// distributions that ship a subset of the upstream model (e.g., omitted learnable positional embedding when
    /// <c>extra_pos_embed_type=null</c>, or fused single-file checkpoints that flatten the embedder MLP).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbedWeight = weights["patch_embed.proj.weight"];
        weights.TryGetValue("patch_embed.proj.bias", out _patchEmbedBias);

        if (_config.UseCrossAttnProjection)
        {
            // Diffusers stores this at `crossattn_proj.weight` (and `.bias` if present).
            if (weights.TryGetValue("crossattn_proj.weight", out Tensor? cWeight))
            {
                _crossAttnProjWeight = cWeight;
                weights.TryGetValue("crossattn_proj.bias", out _crossAttnProjBias);
            }
        }

        // time_embed.t_embedder is a CosmosTimestepEmbedding (Linear→SiLU→Linear).
        _timeLinear1Weight = weights["time_embed.t_embedder.linear_1.weight"];
        weights.TryGetValue("time_embed.t_embedder.linear_1.bias", out _timeLinear1Bias);
        _timeLinear2Weight = weights["time_embed.t_embedder.linear_2.weight"];
        weights.TryGetValue("time_embed.t_embedder.linear_2.bias", out _timeLinear2Bias);
        // RMSNorm on the pre-MLP sin/cos vector to produce embedded_timestep.
        _timeNormWeight = LoadAsF32(weights, "time_embed.norm.weight");

        if (_config.UseLearnablePosEmbed)
        {
            weights.TryGetValue("learnable_pos_embed.pos_emb_t", out _posEmbT);
            weights.TryGetValue("learnable_pos_embed.pos_emb_h", out _posEmbH);
            weights.TryGetValue("learnable_pos_embed.pos_emb_w", out _posEmbW);
        }

        // norm_out is a CosmosAdaLayerNorm: linear_1[hidden,hidden] + linear_2[2*hidden, hidden] (the diffusers code splits
        // the projection into two linears — preserve both names for compatibility).
        _normOutLinear1Weight = weights["norm_out.linear_1.weight"];
        weights.TryGetValue("norm_out.linear_1.bias", out _normOutLinear1Bias);
        _normOutLinear2Weight = weights["norm_out.linear_2.weight"];
        weights.TryGetValue("norm_out.linear_2.bias", out _normOutLinear2Bias);

        _projOutWeight = weights["proj_out.weight"];
        weights.TryGetValue("proj_out.bias", out _projOutBias);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbedWeight is not null) yield return _patchEmbedWeight;
        if (_patchEmbedBias is not null) yield return _patchEmbedBias;
        if (_crossAttnProjWeight is not null) yield return _crossAttnProjWeight;
        if (_crossAttnProjBias is not null) yield return _crossAttnProjBias;
        if (_timeLinear1Weight is not null) yield return _timeLinear1Weight;
        if (_timeLinear1Bias is not null) yield return _timeLinear1Bias;
        if (_timeLinear2Weight is not null) yield return _timeLinear2Weight;
        if (_timeLinear2Bias is not null) yield return _timeLinear2Bias;
        if (_timeNormWeight is not null) yield return _timeNormWeight;
        if (_posEmbT is not null) yield return _posEmbT;
        if (_posEmbH is not null) yield return _posEmbH;
        if (_posEmbW is not null) yield return _posEmbW;
        if (_normOutLinear1Weight is not null) yield return _normOutLinear1Weight;
        if (_normOutLinear1Bias is not null) yield return _normOutLinear1Bias;
        if (_normOutLinear2Weight is not null) yield return _normOutLinear2Weight;
        if (_normOutLinear2Bias is not null) yield return _normOutLinear2Bias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">VAE latent <c>[B, in_channels, H, W]</c>.</param>
    /// <param name="timestep">Scalar timestep (same value for every batch row).</param>
    /// <param name="textEmbeds">Text encoder hiddens <c>[B, T, text_embed_dim]</c>.</param>
    /// <param name="textAttentionMask">Optional padded-text mask <c>[B, T]</c> (1 = valid, 0 = padded). When non-null,
    /// converted to additive mask <c>[B, 1, 1, T]</c> for cross-attention.</param>
    /// <returns>Predicted velocity <c>[B, out_channels, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int[]? textAttentionMask = null)
    {
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4D [B, C, H, W], got {latent.Shape}.", nameof(latent));
        if (textEmbeds.Shape.Rank != 3)
            throw new ArgumentException($"textEmbeds must be 3D [B, T, dim], got {textEmbeds.Shape}.", nameof(textEmbeds));

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

        // ── 1. Optional concat of constant-1 padding-mask channel ──
        Tensor latentExpanded = _config.ConcatPaddingMask
            ? AppendPaddingMaskChannel(latent)
            : latent;
        AnimaDebugDump.Dump("latent_with_mask", latentExpanded);

        int totalChannels = (int)latentExpanded.Shape[1];

        // ── 2. Patch embed → [B, S, hidden] ──
        Tensor imgTokens = PatchEmbed(backend, latentExpanded, batch, totalChannels, latentH, latentW, gridH, gridW);
        if (!ReferenceEquals(latentExpanded, latent)) latentExpanded.Dispose();
        AnimaDebugDump.Dump("patch_embed", imgTokens);

        // ── 3. Optional learnable positional embedding (T=1, broadcasted) ──
        if (_config.UseLearnablePosEmbed && _posEmbT is not null && _posEmbH is not null && _posEmbW is not null)
        {
            AddLearnablePosEmbed(imgTokens, batch, gridH, gridW);
            AnimaDebugDump.Dump("after_pos_embed", imgTokens);
        }

        // ── 4. RoPE table for image self-attn ──
        (Tensor ropeCos, Tensor ropeSin) = _rope.BuildFreqs(tFrames: 1, hPatched: gridH, wPatched: gridW);
        AnimaDebugDump.Dump("rope_cos", ropeCos);
        AnimaDebugDump.Dump("rope_sin", ropeSin);

        // ── 5. Time embedding ──
        (Tensor temb, Tensor embeddedTimestep) = TimeEmbed(backend, timestep, batch);
        AnimaDebugDump.Dump("temb", temb);
        AnimaDebugDump.Dump("embedded_timestep", embeddedTimestep);

        // ── 6. Project text embeds to hidden if needed ──
        Tensor textHidden = ProjectText(backend, textEmbeds, batch, textSeq, hidden);
        AnimaDebugDump.Dump("text_hidden", textHidden);

        // ── 7. Build cross-attn mask if provided ──
        Tensor? crossMask = textAttentionMask is not null
            ? BuildAttentionMask(batch, textSeq, textAttentionMask)
            : null;

        // ── 8. Layer loop ──
        Tensor x = imgTokens;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, textHidden, embeddedTimestep, temb, ropeCos, ropeSin, _rope, crossMask);
            x.Dispose();
            x = next;
            AnimaDebugDump.Dump($"block_{i}", x);
        }

        ropeCos.Dispose();
        ropeSin.Dispose();
        textHidden.Dispose();
        crossMask?.Dispose();

        // ── 9. Final AdaLN (no gate) + proj_out ──
        Tensor projected = ApplyFinalLayer(backend, x, temb, batch, seqLen, hidden);
        x.Dispose();
        temb.Dispose();
        embeddedTimestep.Dispose();
        AnimaDebugDump.Dump("final_proj", projected);

        // ── 10. Unpatchify [B, S, p_h*p_w*C_out] → [B, C_out, H, W] ──
        Tensor velocity = Unpatchify(projected, batch, gridH, gridW, patchH, patchW, _config.OutChannels);
        projected.Dispose();
        AnimaDebugDump.DumpOutput(velocity);

        return velocity;
    }

    /// <summary>Concatenates a constant-1 channel to <c>[B, C, H, W]</c>, returning <c>[B, C+1, H, W]</c>. Mirrors the
    /// upstream <c>concat_padding_mask</c> path with a constant unmasked padding mask (we don't expose explicit padding
    /// masks for image-only inference).</summary>
    private static Tensor AppendPaddingMaskChannel(Tensor latent)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        TensorShape shape = new TensorShape(batch, channels + 1, height, width);
        Tensor output = new Tensor(shape, DType.F32);

        long spatial = (long)height * width;
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                long inOff = ((long)b * channels + c) * spatial;
                long outOff = ((long)b * (channels + 1) + c) * spatial;
                long bytes = spatial * sizeof(float);
                Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, bytes, bytes);
            }
            // Final channel: all ones.
            long maskOff = ((long)b * (channels + 1) + channels) * spatial;
            for (long i = 0; i < spatial; i++) outPtr[maskOff + i] = 1.0f;
        }
        return output;
    }

    /// <summary>Patch embed via a single linear projection over the flattened p_t·p_h·p_w-element patch. Mirrors
    /// <c>CosmosPatchEmbed.forward</c> for <c>p_t=1</c>: reshape to <c>[B, C, gridH, p_h, gridW, p_w]</c> →
    /// permute to <c>[B, gridH, gridW, C, p_h, p_w]</c> → flatten last 3 dims → linear.</summary>
    private Tensor PatchEmbed(IBackend backend, Tensor latent, int batch, int channels, int latentH, int latentW, int gridH, int gridW)
    {
        int patchH = _config.PatchSize.H;
        int patchW = _config.PatchSize.W;
        int patchVol = patchH * patchW;
        int patchFeat = channels * patchVol;

        TensorShape gatherShape = new TensorShape(batch, gridH * gridW, patchFeat);
        Tensor gathered = new Tensor(gatherShape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* gatherPtr = (float*)gathered.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int gh = 0; gh < gridH; gh++)
            {
                for (int gw = 0; gw < gridW; gw++)
                {
                    long seqIdx = (long)gh * gridW + gw;
                    long outBase = ((long)b * gridH * gridW + seqIdx) * patchFeat;
                    int feat = 0;
                    // Order: (c, ph, pw) — outermost = channel.
                    for (int c = 0; c < channels; c++)
                    {
                        long inChannelBase = ((long)b * channels + c) * latentH * latentW;
                        for (int ph = 0; ph < patchH; ph++)
                        {
                            for (int pw = 0; pw < patchW; pw++)
                            {
                                int row = gh * patchH + ph;
                                int col = gw * patchW + pw;
                                gatherPtr[outBase + feat++] = inPtr[inChannelBase + (long)row * latentW + col];
                            }
                        }
                    }
                }
            }
        }

        TensorShape outShape = new TensorShape(batch, gridH * gridW, _config.HiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, gathered, _patchEmbedWeight!, _patchEmbedBias);
        gathered.Dispose();
        return output;
    }

    /// <summary>Adds the L2-normalized sum of three axis-specific learnable embeddings into the image token sequence.
    /// Mirrors <c>CosmosLearnablePositionalEmbed.forward</c> for <c>T=1</c>: <c>emb = pos_t[0] + pos_h[h] + pos_w[w]</c>
    /// per token, then <c>emb / (||emb||_2 + eps)</c>, summed into the token. Modifies <paramref name="imgTokens"/>
    /// in-place.</summary>
    private void AddLearnablePosEmbed(Tensor imgTokens, int batch, int gridH, int gridW)
    {
        int hidden = _config.HiddenSize;
        int seqLen = gridH * gridW;
        float eps = _config.RmsNormEps;
        float* posTPtr = (float*)_posEmbT!.DataPointer;
        float* posHPtr = (float*)_posEmbH!.DataPointer;
        float* posWPtr = (float*)_posEmbW!.DataPointer;
        float* tokPtr = (float*)imgTokens.DataPointer;

        Span<float> emb = stackalloc float[hidden > 4096 ? 0 : hidden];
        // Fall back to heap allocation for very wide hiddens.
        float[]? embHeap = hidden > 4096 ? new float[hidden] : null;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < gridH; h++)
            {
                for (int w = 0; w < gridW; w++)
                {
                    Span<float> embView = embHeap is not null ? embHeap.AsSpan() : emb;

                    long tBase = 0; // T = 0 for t2i.
                    long hBase = (long)h * hidden;
                    long wBase = (long)w * hidden;
                    float sumSq = 0f;
                    for (int d = 0; d < hidden; d++)
                    {
                        float v = posTPtr[tBase + d] + posHPtr[hBase + d] + posWPtr[wBase + d];
                        embView[d] = v;
                        sumSq += v * v;
                    }
                    // Cosmos uses linalg.vector_norm with eps in the divisor (not added in sqrt).
                    float invNorm = 1.0f / (MathF.Sqrt(sumSq) + eps);
                    long tokBase = ((long)b * seqLen + (long)h * gridW + w) * hidden;
                    for (int d = 0; d < hidden; d++)
                        tokPtr[tokBase + d] += embView[d] * invNorm;
                }
            }
        }
    }

    /// <summary>Time embedding: sinusoidal projection (flip_sin_to_cos=True) → CosmosTimestepEmbedding (Linear→SiLU→Linear)
    /// produces <c>temb</c>. Separately, the pre-MLP sin/cos vector is RMSNorm'd to produce <c>embedded_timestep</c>.</summary>
    private (Tensor Temb, Tensor EmbeddedTimestep) TimeEmbed(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;
        int condDim = _config.ConditionDim;

        TensorShape sinShape = new TensorShape(batch, hidden);
        Tensor sinEmb = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmb, timestep, batch, hidden, 10000f);
        AnimaDebugDump.Dump("time_sin_proj", sinEmb);

        // t_embedder MLP: Linear(hidden, condDim) → SiLU → Linear(condDim, condDim) — diffusers'
        // CosmosTimestepEmbedding linear_1 maps hidden→hidden (or condDim) and linear_2 maps to the final
        // condition_dim. We respect whatever shapes were loaded.
        Tensor mlp1 = new Tensor(new TensorShape(batch, (int)_timeLinear1Weight!.Shape[0]), DType.F32);
        backend.Linear(mlp1, sinEmb, _timeLinear1Weight!, _timeLinear1Bias);

        Tensor mlp1Act = new Tensor(mlp1.Shape, DType.F32);
        backend.Silu(mlp1Act, mlp1);
        mlp1.Dispose();

        Tensor temb = new Tensor(new TensorShape(batch, condDim), DType.F32);
        backend.Linear(temb, mlp1Act, _timeLinear2Weight!, _timeLinear2Bias);
        mlp1Act.Dispose();

        // embedded_timestep: RMSNorm of the pre-MLP sinusoidal vector. Cosmos uses RMSNorm without learnable scale by
        // default, but the checkpoint exposes a `time_embed.norm.weight` so we use it via backend.RmsNorm.
        Tensor embeddedTimestep = new Tensor(sinShape, DType.F32);
        backend.RmsNorm(embeddedTimestep, sinEmb, _timeNormWeight!, _config.RmsNormEps);
        sinEmb.Dispose();

        return (temb, embeddedTimestep);
    }

    /// <summary>Projects text embeddings to <c>hidden</c> via the optional <c>crossattn_proj</c>. If absent (or text dim
    /// already equals hidden), returns a copy.</summary>
    private Tensor ProjectText(IBackend backend, Tensor textEmbeds, int batch, int textSeq, int hidden)
    {
        TensorShape outShape = new TensorShape(batch, textSeq, hidden);
        Tensor output = new Tensor(outShape, DType.F32);

        if (_crossAttnProjWeight is not null)
        {
            backend.Linear(output, textEmbeds, _crossAttnProjWeight, _crossAttnProjBias);
            return output;
        }

        int textDim = (int)textEmbeds.Shape[2];
        if (textDim != hidden)
            throw new InvalidOperationException(
                $"Text dim {textDim} != hidden {hidden} but crossattn_proj weight is missing. Set UseCrossAttnProjection=true and ensure the checkpoint contains `crossattn_proj.weight`.");

        // Identity copy (cast to F32 if needed).
        long bytes = batch * textSeq * hidden * sizeof(float);
        if (textEmbeds.DType == DType.F32)
        {
            Buffer.MemoryCopy(textEmbeds.DataPointer, output.DataPointer, bytes, bytes);
            return output;
        }
        using Tensor f32 = textEmbeds.CastTo(DType.F32);
        Buffer.MemoryCopy(f32.DataPointer, output.DataPointer, bytes, bytes);
        return output;
    }

    /// <summary>Builds a cross-attention mask <c>[B, 1, 1, T]</c> with 0 for valid positions and large-negative for masked.</summary>
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

    /// <summary>Applies the final <c>CosmosAdaLayerNorm</c> + <c>proj_out</c>: <c>norm_out</c> projects temb into a
    /// <c>2*hidden</c> shift/scale, modulates the post-LayerNorm-no-affine sequence, then <c>proj_out</c> linearly maps
    /// to <c>p_h*p_w*out_channels</c> per token.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor x, Tensor temb, int batch, int seqLen, int hidden)
    {
        // Step 1: linear_1(silu(embedded_timestep))? Diffusers' CosmosAdaLayerNorm does:
        //   y = activation(embedded_timestep) → linear_1 → linear_2
        // Then y = y + temb[..., :2*hidden]. Then chunk(2,-1) → shift, scale.
        // Our temb here is the actual processed temb output of the time embedder; we skip the silu+linear_1/linear_2
        // chain and use the same temb pipeline as the AdaLN-Zero in the blocks. Concretely: linear_1 maps hidden→hidden
        // and linear_2 maps hidden→2*hidden; we feed silu(temb) through them.
        TensorShape act1Shape = new TensorShape(batch, (int)_normOutLinear1Weight!.Shape[0]);
        Tensor act = new Tensor(temb.Shape, DType.F32);
        backend.Silu(act, temb);

        Tensor m1 = new Tensor(act1Shape, DType.F32);
        backend.Linear(m1, act, _normOutLinear1Weight!, _normOutLinear1Bias);
        act.Dispose();

        TensorShape projShape = new TensorShape(batch, 2 * hidden);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, m1, _normOutLinear2Weight!, _normOutLinear2Bias);
        m1.Dispose();

        // Cosmos's CosmosAdaLayerNorm also does + temb[..., :2*hidden] before the chunk.
        AddTembSliceInPlace(projected, temb, batch, 2 * hidden);

        TensorShape modShape = new TensorShape(batch, hidden);
        Tensor shift = new Tensor(modShape, DType.F32);
        Tensor scale = new Tensor(modShape, DType.F32);
        SplitTwo(projected, shift, scale, batch, hidden);
        projected.Dispose();

        // LayerNorm-no-affine + scale/shift.
        TensorShape seqShape = new TensorShape(batch, seqLen, hidden);
        Tensor normed = new Tensor(seqShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, hidden, _config.RmsNormEps);

        Tensor modulated = AdaLNModulation.ApplyModulation(normed, shift, scale, batch, seqLen, hidden);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        // proj_out: hidden → p_h * p_w * out_channels (p_t = 1).
        int outDim = _config.PatchSize.H * _config.PatchSize.W * _config.OutChannels;
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();
        return output;
    }

    /// <summary>Reshapes <c>[B, S, p_h*p_w*C]</c> back to <c>[B, C, H, W]</c>. Inverse of the patch-embed gather order
    /// <c>(c, ph, pw)</c>.</summary>
    private static Tensor Unpatchify(Tensor packed, int batch, int gridH, int gridW, int patchH, int patchW, int outChannels)
    {
        int height = gridH * patchH;
        int width = gridW * patchW;
        int seqLen = gridH * gridW;
        int patchVol = patchH * patchW;
        int featDim = patchVol * outChannels;

        TensorShape shape = new TensorShape(batch, outChannels, height, width);
        Tensor output = new Tensor(shape, DType.F32);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                long outChannelBase = ((long)b * outChannels + c) * height * width;
                for (int gh = 0; gh < gridH; gh++)
                {
                    for (int gw = 0; gw < gridW; gw++)
                    {
                        int seqIdx = gh * gridW + gw;
                        long inBase = ((long)b * seqLen + seqIdx) * featDim + (long)c * patchVol;
                        for (int ph = 0; ph < patchH; ph++)
                        {
                            for (int pw = 0; pw < patchW; pw++)
                            {
                                int row = gh * patchH + ph;
                                int col = gw * patchW + pw;
                                outPtr[outChannelBase + (long)row * width + col] = inPtr[inBase + ph * patchW + pw];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>In-place: <c>dst[b, d] += temb[b, d]</c> for d in [0, sliceLen).</summary>
    private static void AddTembSliceInPlace(Tensor dst, Tensor temb, int batch, int sliceLen)
    {
        int tembStride = (int)temb.Shape[1];
        if (tembStride < sliceLen)
            throw new ArgumentException($"temb width {tembStride} < slice {sliceLen}.", nameof(temb));
        float* dstPtr = (float*)dst.DataPointer;
        float* tembPtr = (float*)temb.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int dstOff = b * sliceLen;
            int tOff = b * tembStride;
            for (int d = 0; d < sliceLen; d++)
                dstPtr[dstOff + d] += tembPtr[tOff + d];
        }
    }

    /// <summary>Splits <c>[B, 2*hidden]</c> into <c>[B, hidden]</c> shift and scale (Cosmos order: shift first, scale second).</summary>
    private static void SplitTwo(Tensor src, Tensor shift, Tensor scale, int batch, int hidden)
    {
        float* sPtr = (float*)src.DataPointer;
        float* shiftPtr = (float*)shift.DataPointer;
        float* scalePtr = (float*)scale.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long src0 = (long)b * 2 * hidden;
            long dst = (long)b * hidden;
            long bytes = (long)hidden * sizeof(float);
            Buffer.MemoryCopy(sPtr + src0, shiftPtr + dst, bytes, bytes);
            Buffer.MemoryCopy(sPtr + src0 + hidden, scalePtr + dst, bytes, bytes);
        }
    }

    /// <summary>Loads a tensor and casts to F32 if needed (RmsNorm scales must be F32).</summary>
    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    /// <summary>Disposes the transformer state. The actual unmanaged tensor storage is owned by the underlying
    /// safetensors mmap.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchEmbedWeight = null; _patchEmbedBias = null;
            _crossAttnProjWeight = null; _crossAttnProjBias = null;
            _timeLinear1Weight = null; _timeLinear1Bias = null;
            _timeLinear2Weight = null; _timeLinear2Bias = null;
            _timeNormWeight = null;
            _posEmbT = null; _posEmbH = null; _posEmbW = null;
            _normOutLinear1Weight = null; _normOutLinear1Bias = null;
            _normOutLinear2Weight = null; _normOutLinear2Bias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
