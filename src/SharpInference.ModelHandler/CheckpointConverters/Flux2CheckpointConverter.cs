using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>
/// Converts BFL-format Flux.2 (Klein 4B / Klein 9B / Dev) checkpoints to the canonical layout
/// loaded by <see cref="Flux2Transformer"/>. Per-block remapping mirrors
/// <see cref="FluxCheckpointConverter"/> but accounts for Flux.2-specific changes:
/// <list type="bullet">
/// <item>Top-level shared modulation projections: <c>double_stream_modulation_img/txt.lin</c> and
/// <c>single_stream_modulation.lin</c> rename to <c>double_stream_modulation_img.linear</c> /
/// <c>single_stream_modulation.linear</c> (per-block <c>img_mod</c>/<c>txt_mod</c> do not exist
/// in Flux.2).</item>
/// <item>Double-block MLP <c>img_mlp.0.weight: [2*mlp_inner, hidden]</c> is the fused SwiGLU
/// <c>linear_in</c>; we split into <c>ff.linear_in_gate</c> (rows 0..mlp_inner) and
/// <c>ff.linear_in_up</c> (rows mlp_inner..2*mlp_inner). <c>img_mlp.2</c> renames to
/// <c>ff.linear_out</c>. Same for txt-stream → <c>ff_context.*</c>.</item>
/// <item>Single-block <c>linear1</c> packs <c>Q + K + V + gate + up</c> (3*hidden + 2*mlp_inner
/// rows); we split into 5 separate weights. <c>linear2</c> is kept fused and renamed to
/// <c>attn.to_out</c>.</item>
/// <item>Final-layer <c>norm_out.linear</c> needs <c>swap_scale_shift</c> on its first dim
/// (BFL stores [shift, scale] but the diffusers convention used by
/// <see cref="Flux2Transformer"/>'s final layer is [scale, shift]).</item>
/// <item>No CLIP pooled MLP and no <c>guidance_in</c> on Klein; Dev has <c>guidance_in</c>.</item>
/// </list>
/// </summary>
public sealed class Flux2CheckpointConverter
{
    /// <summary>Result of converting a single-file Flux.2 checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    private readonly int _hiddenSize;
    private readonly int _mlpInner;

    /// <summary>Creates a converter sized for a specific Flux.2 variant.</summary>
    /// <param name="hiddenSize">Transformer hidden dim. 3072 for Klein 4B, 4096 for Klein 9B, 6144 for Dev.</param>
    /// <param name="mlpInner">Per-block MLP inner dimension (= hiddenSize × mlp_ratio). 9216 for Klein 4B (ratio 3.0), etc.</param>
    public Flux2CheckpointConverter(int hiddenSize, int mlpInner)
    {
        _hiddenSize = hiddenSize;
        _mlpInner = mlpInner;
    }

    /// <summary>Converts a transformer-only single-file Flux.2 checkpoint (BFL format) into the canonical key layout consumed by <see cref="Flux2Transformer"/>.</summary>
    public Dictionary<string, Tensor> ConvertTransformer(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> output = new(allWeights.Count + 100);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            // Strip optional BFL prefix
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                key = key["model.diffusion_model.".Length..];

            ConvertKey(key, tensor, output);
        }

        return output;
    }

    private void ConvertKey(string bflKey, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Top-level renames
        if (bflKey.StartsWith("img_in.", StringComparison.Ordinal))
        {
            output["x_embedder." + bflKey["img_in.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("txt_in.", StringComparison.Ordinal))
        {
            output["context_embedder." + bflKey["txt_in.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("time_in.in_layer.", StringComparison.Ordinal))
        {
            output["time_guidance_embed.timestep_embedder.linear_1." + bflKey["time_in.in_layer.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("time_in.out_layer.", StringComparison.Ordinal))
        {
            output["time_guidance_embed.timestep_embedder.linear_2." + bflKey["time_in.out_layer.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("guidance_in.in_layer.", StringComparison.Ordinal))
        {
            output["time_guidance_embed.guidance_embedder.linear_1." + bflKey["guidance_in.in_layer.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("guidance_in.out_layer.", StringComparison.Ordinal))
        {
            output["time_guidance_embed.guidance_embedder.linear_2." + bflKey["guidance_in.out_layer.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("double_stream_modulation_img.lin.", StringComparison.Ordinal))
        {
            output["double_stream_modulation_img.linear." + bflKey["double_stream_modulation_img.lin.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("double_stream_modulation_txt.lin.", StringComparison.Ordinal))
        {
            output["double_stream_modulation_txt.linear." + bflKey["double_stream_modulation_txt.lin.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("single_stream_modulation.lin.", StringComparison.Ordinal))
        {
            output["single_stream_modulation.linear." + bflKey["single_stream_modulation.lin.".Length..]] = tensor;
            return;
        }
        if (bflKey.StartsWith("final_layer.", StringComparison.Ordinal))
        {
            ConvertFinalLayer(bflKey["final_layer.".Length..], tensor, output);
            return;
        }
        if (bflKey.StartsWith("double_blocks.", StringComparison.Ordinal))
        {
            ConvertDoubleBlock(bflKey["double_blocks.".Length..], tensor, output);
            return;
        }
        if (bflKey.StartsWith("single_blocks.", StringComparison.Ordinal))
        {
            ConvertSingleBlock(bflKey["single_blocks.".Length..], tensor, output);
            return;
        }
        // Unknown keys are silently dropped (allows the same checkpoint to carry text-encoder
        // weights without confusing the transformer load).
    }

    private static void ConvertFinalLayer(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest.StartsWith("adaLN_modulation.1.", StringComparison.Ordinal))
        {
            // BFL stores [shift, scale] (rows 0..H = shift, H..2H = scale); the Flux.2 transformer's
            // final layer matches the diffusers convention [scale, shift]. Swap the halves to
            // align — same fix as Flux.1 deviation #23.
            Tensor swapped = SwapScaleShiftHalves(tensor);
            output["norm_out.linear." + rest["adaLN_modulation.1.".Length..]] = swapped;
            return;
        }
        if (rest.StartsWith("linear.", StringComparison.Ordinal))
        {
            output["proj_out." + rest["linear.".Length..]] = tensor;
            return;
        }
    }

    private void ConvertDoubleBlock(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string blockIdxStr = rest[..firstDot];
        string after = rest[(firstDot + 1)..];
        string prefix = $"transformer_blocks.{blockIdxStr}";

        if (after.StartsWith("img_attn.", StringComparison.Ordinal))
        {
            ConvertDoubleAttn(prefix, "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.norm_q", "attn.norm_k",
                after["img_attn.".Length..], tensor, output);
            return;
        }
        if (after.StartsWith("txt_attn.", StringComparison.Ordinal))
        {
            ConvertDoubleAttn(prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "attn.norm_added_q", "attn.norm_added_k",
                after["txt_attn.".Length..], tensor, output);
            return;
        }
        if (after.StartsWith("img_mlp.", StringComparison.Ordinal))
        {
            ConvertDoubleMlp(prefix, "ff", after["img_mlp.".Length..], tensor, output);
            return;
        }
        if (after.StartsWith("txt_mlp.", StringComparison.Ordinal))
        {
            ConvertDoubleMlp(prefix, "ff_context", after["txt_mlp.".Length..], tensor, output);
            return;
        }
    }

    private void ConvertDoubleAttn(string prefix, string qName, string kName, string vName, string outName,
        string normQName, string normKName, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest == "qkv.weight")
        {
            SplitQkvWeight(tensor, _hiddenSize, prefix, qName, kName, vName, output);
            return;
        }
        if (rest == "qkv.bias")
        {
            SplitQkvBias(tensor, _hiddenSize, prefix, qName, kName, vName, output);
            return;
        }
        if (rest.StartsWith("proj.", StringComparison.Ordinal))
        {
            output[$"{prefix}.{outName}.{rest["proj.".Length..]}"] = tensor;
            return;
        }
        // QK-norm — only "norm.query_norm.scale" / "norm.key_norm.scale" naming exists in the BFL Flux.2 checkpoint.
        if (rest == "norm.query_norm.scale" || rest == "norm_q.weight")
        {
            output[$"{prefix}.{normQName}.weight"] = tensor;
            return;
        }
        if (rest == "norm.key_norm.scale" || rest == "norm_k.weight")
        {
            output[$"{prefix}.{normKName}.weight"] = tensor;
            return;
        }
    }

    /// <summary>
    /// Splits the fused SwiGLU <c>linear_in</c> in a Flux.2 double block's MLP into separate gate
    /// and up projections, and renames <c>*_mlp.2</c> to <c>linear_out</c>.
    /// </summary>
    private void ConvertDoubleMlp(string prefix, string ffName, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest == "0.weight")
        {
            SplitSwiGluLinearIn(tensor, prefix, ffName, output, hasBias: false);
            return;
        }
        if (rest == "0.bias")
        {
            SplitSwiGluLinearIn(tensor, prefix, ffName, output, hasBias: true);
            return;
        }
        if (rest.StartsWith("2.", StringComparison.Ordinal))
        {
            output[$"{prefix}.{ffName}.linear_out.{rest[2..]}"] = tensor;
            return;
        }
    }

    private void ConvertSingleBlock(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string blockIdxStr = rest[..firstDot];
        string after = rest[(firstDot + 1)..];
        string prefix = $"single_transformer_blocks.{blockIdxStr}";

        if (after == "linear1.weight")
        {
            SplitSingleLinear1Weight(tensor, prefix, output);
            return;
        }
        if (after == "linear1.bias")
        {
            SplitSingleLinear1Bias(tensor, prefix, output);
            return;
        }
        if (after.StartsWith("linear2.", StringComparison.Ordinal))
        {
            // Fused output: [hidden + mlp_inner → hidden]. Kept whole for use by Flux2SingleBlock.
            output[$"{prefix}.attn.to_out.{after["linear2.".Length..]}"] = tensor;
            return;
        }
        if (after == "norm.query_norm.scale" || after == "norm_q.weight")
        {
            output[$"{prefix}.attn.norm_q.weight"] = tensor;
            return;
        }
        if (after == "norm.key_norm.scale" || after == "norm_k.weight")
        {
            output[$"{prefix}.attn.norm_k.weight"] = tensor;
            return;
        }
    }

    // ── Static helpers ──

    private static unsafe Tensor SwapScaleShiftHalves(Tensor input)
    {
        long firstDim = input.Shape[0];
        if (firstDim % 2 != 0)
            throw new InvalidOperationException(
                $"SwapScaleShiftHalves: first dim must be even, got {firstDim}");

        long totalElements = input.ElementCount;
        long elemBytes = input.DType.SizeInBytes;
        long halfBytes = (totalElements / 2) * elemBytes;

        Tensor swapped = new Tensor(input.Shape, input.DType);
        swapped.Fp8ScaleFactor = input.Fp8ScaleFactor;

        byte* src = (byte*)input.DataPointer;
        byte* dst = (byte*)swapped.DataPointer;
        Buffer.MemoryCopy(src + halfBytes, dst, halfBytes, halfBytes);
        Buffer.MemoryCopy(src, dst + halfBytes, halfBytes, halfBytes);
        return swapped;
    }

    private static unsafe void SplitQkvWeight(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = (long)inDim * fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * rowBytes;
        TensorShape splitShape = new TensorShape(innerDim, inDim);

        Tensor qWeight = new Tensor(splitShape, fused.DType);
        Tensor kWeight = new Tensor(splitShape, fused.DType);
        Tensor vWeight = new Tensor(splitShape, fused.DType);
        qWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vWeight.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.weight"] = qWeight;
        output[$"{prefix}.{kName}.weight"] = kWeight;
        output[$"{prefix}.{vName}.weight"] = vWeight;
    }

    private static unsafe void SplitQkvBias(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        long elemBytes = fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * elemBytes;
        TensorShape splitShape = new TensorShape(innerDim);

        Tensor qBias = new Tensor(splitShape, fused.DType);
        Tensor kBias = new Tensor(splitShape, fused.DType);
        Tensor vBias = new Tensor(splitShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vBias.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.bias"] = qBias;
        output[$"{prefix}.{kName}.bias"] = kBias;
        output[$"{prefix}.{vName}.bias"] = vBias;
    }

    /// <summary>
    /// Splits a SwiGLU <c>linear_in</c> weight or bias along dim 0 into <c>linear_in_gate</c>
    /// (first half — the gate proj that gets <c>silu</c>) and <c>linear_in_up</c> (second half —
    /// the up proj that multiplies the gated activation).
    /// </summary>
    private unsafe void SplitSwiGluLinearIn(Tensor fused, string prefix, string ffName,
        Dictionary<string, Tensor> output, bool hasBias)
    {
        long firstDim = fused.Shape[0];
        if (firstDim != 2L * _mlpInner)
            throw new InvalidOperationException(
                $"SplitSwiGluLinearIn: expected first dim = 2 * mlp_inner ({2 * _mlpInner}), got {firstDim} for {prefix}.{ffName}");
        long totalElements = fused.ElementCount;
        long elemBytes = fused.DType.SizeInBytes;
        long halfBytes = (totalElements / 2) * elemBytes;

        TensorShape halfShape = hasBias
            ? new TensorShape(_mlpInner)
            : new TensorShape(_mlpInner, fused.Shape[1]);

        Tensor gate = new Tensor(halfShape, fused.DType);
        Tensor up = new Tensor(halfShape, fused.DType);
        gate.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        up.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)gate.DataPointer, halfBytes, halfBytes);
        Buffer.MemoryCopy(src + halfBytes, (void*)up.DataPointer, halfBytes, halfBytes);

        string suffix = hasBias ? "bias" : "weight";
        output[$"{prefix}.{ffName}.linear_in_gate.{suffix}"] = gate;
        output[$"{prefix}.{ffName}.linear_in_up.{suffix}"] = up;
    }

    /// <summary>
    /// Splits the fused single-block <c>linear1</c> weight <c>[3*hidden + 2*mlp_inner, hidden]</c>
    /// into <c>attn.to_{q,k,v}</c> (each <c>[hidden, hidden]</c>) plus <c>attn.linear_in_{gate,up}</c>
    /// (each <c>[mlp_inner, hidden]</c>).
    /// </summary>
    private unsafe void SplitSingleLinear1Weight(Tensor fused, string prefix, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = (long)inDim * fused.DType.SizeInBytes;

        TensorShape qkvShape = new TensorShape(_hiddenSize, inDim);
        TensorShape mlpShape = new TensorShape(_mlpInner, inDim);

        Tensor q = new Tensor(qkvShape, fused.DType);
        Tensor k = new Tensor(qkvShape, fused.DType);
        Tensor v = new Tensor(qkvShape, fused.DType);
        Tensor gate = new Tensor(mlpShape, fused.DType);
        Tensor up = new Tensor(mlpShape, fused.DType);

        q.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        k.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        v.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        gate.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        up.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = (long)_hiddenSize * rowBytes;
        long mlpChunkBytes = (long)_mlpInner * rowBytes;

        long offset = 0;
        Buffer.MemoryCopy(src + offset, (void*)q.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)k.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)v.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)gate.DataPointer, mlpChunkBytes, mlpChunkBytes); offset += mlpChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)up.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{prefix}.attn.to_q.weight"] = q;
        output[$"{prefix}.attn.to_k.weight"] = k;
        output[$"{prefix}.attn.to_v.weight"] = v;
        output[$"{prefix}.attn.linear_in_gate.weight"] = gate;
        output[$"{prefix}.attn.linear_in_up.weight"] = up;
    }

    private unsafe void SplitSingleLinear1Bias(Tensor fused, string prefix, Dictionary<string, Tensor> output)
    {
        long elemBytes = fused.DType.SizeInBytes;
        TensorShape qkvShape = new TensorShape(_hiddenSize);
        TensorShape mlpShape = new TensorShape(_mlpInner);

        Tensor q = new Tensor(qkvShape, fused.DType);
        Tensor k = new Tensor(qkvShape, fused.DType);
        Tensor v = new Tensor(qkvShape, fused.DType);
        Tensor gate = new Tensor(mlpShape, fused.DType);
        Tensor up = new Tensor(mlpShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = (long)_hiddenSize * elemBytes;
        long mlpChunkBytes = (long)_mlpInner * elemBytes;

        long offset = 0;
        Buffer.MemoryCopy(src + offset, (void*)q.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)k.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)v.DataPointer, qkvChunkBytes, qkvChunkBytes); offset += qkvChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)gate.DataPointer, mlpChunkBytes, mlpChunkBytes); offset += mlpChunkBytes;
        Buffer.MemoryCopy(src + offset, (void*)up.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{prefix}.attn.to_q.bias"] = q;
        output[$"{prefix}.attn.to_k.bias"] = k;
        output[$"{prefix}.attn.to_v.bias"] = v;
        output[$"{prefix}.attn.linear_in_gate.bias"] = gate;
        output[$"{prefix}.attn.linear_in_up.bias"] = up;
    }
}
