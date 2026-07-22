using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a HunyuanVideo DiT checkpoint into the "hybrid" key naming
/// <see cref="HartsyInference.Diffusion.Models.Denoisers.HunyuanVideoDit"/> expects — original Tencent head keys
/// (<c>img_in</c>, <c>time_in.0/2</c>, <c>vector_in.0/2</c>, <c>guidance_in.0/2</c>, <c>txt_in.*</c>,
/// <c>final_layer.mod/proj</c>) with the reused <see cref="HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.HunyuanImageBlock"/>
/// diffusers block sub-keys (<c>double_blocks.{i}.norm1.linear</c>, <c>attn.to_q</c>, <c>ff.net.0.proj</c>, …).
///
/// <para>Two source layouts:
/// (1) <b>Tencent/Comfy single-file</b> (the common Comfy-Org repacked <c>hunyuan_video_t2v_720p_*.safetensors</c>):
/// Flux-style names — <c>img_in.proj</c> (Conv3d, reshaped to a 2D patch-embed Linear), <c>double_blocks.*.img_attn.qkv</c>
/// (fused, split into Q/K/V), <c>single_blocks.*.linear1</c> (split into Q/K/V + <c>proj_mlp</c>),
/// <c>*_mod.lin</c>/<c>modulation.lin</c>, <c>img_mlp.0/2</c>, <c>final_layer.linear</c>/<c>adaLN_modulation.1</c>.
/// The token refiner <c>txt_in.*</c> passes through 1:1 (the refiner reads Comfy naming directly).
/// (2) <b>diffusers folder</b> (<c>hunyuanvideo-community/HunyuanVideo transformer/</c>): <c>x_embedder</c>,
/// <c>context_embedder</c>, <c>time_text_embed</c>, <c>transformer_blocks</c>/<c>single_transformer_blocks</c>,
/// <c>norm_out.linear</c> (chunked <c>[scale, shift]</c> — <b>swapped</b> to the Tencent <c>[shift, scale]</c> the DiT
/// reads), <c>proj_out</c>. Block sub-keys map through unchanged; the refiner's split Q/K/V are re-fused to
/// <c>self_attn.qkv</c>.</para>
///
/// <para>fp8 <c>_scaled</c> checkpoints are dequantized first via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>
/// (same as LTX/Flux). The VAE + text encoders ship separately and are not handled here.</para></summary>
public static unsafe class HunyuanVideoCheckpointConverter
{
    // Bundling prefixes seen in the wild: diffusers-style `model.diffusion_model.` and the Comfy-Org repacked
    // single-file's `model.model.` (hunyuan_video_t2v_720p_bf16 wraps everything under model.model.*).
    private static readonly string[] DmPrefixes = ["model.diffusion_model.", "model.model."];

    /// <summary>Converts a flat DiT weight dictionary into the hybrid naming. Sliced/reshaped/re-fused tensors are
    /// newly allocated (owned by the returned dict); pass-through tensors reference the source (keep the loader alive).</summary>
    public static Dictionary<string, Tensor> Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        // Strip a bundling prefix if present (whichever the checkpoint uses).
        string? prefix = null;
        foreach (string p in DmPrefixes)
        {
            foreach (string k in allWeights.Keys) if (k.StartsWith(p, StringComparison.Ordinal)) { prefix = p; break; }
            if (prefix is not null) break;
        }
        if (prefix is not null)
        {
            Dictionary<string, Tensor> stripped = new(allWeights.Count);
            foreach ((string k, Tensor v) in allWeights)
                stripped[k.StartsWith(prefix, StringComparison.Ordinal) ? k[prefix.Length..] : k] = v;
            allWeights = stripped;
        }

        // Raw Tencent hyvideo naming (Kijai fp8 repacks): underscore attention modules (img_attn_qkv),
        // mlp.fc1/fc2, mod.linear — normalize to the Comfy Flux-style names MapOriginal expects.
        bool tencentRaw = allWeights.ContainsKey("double_blocks.0.img_attn_qkv.weight");
        if (tencentRaw)
        {
            Dictionary<string, Tensor> normalized = new(allWeights.Count);
            foreach ((string k, Tensor v) in allWeights) normalized[NormalizeTencentRaw(k)] = v;
            allWeights = normalized;
        }

        bool diffusers = allWeights.ContainsKey("x_embedder.proj.weight")
            || allWeights.ContainsKey("transformer_blocks.0.attn.to_q.weight");
        Dictionary<string, Tensor> outW = new(allWeights.Count + 256);
        foreach ((string key, Tensor t) in allWeights)
        {
            if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8") continue;
            if (diffusers) MapDiffusers(key, t, outW);
            else MapOriginal(key, t, outW);
        }
        return outW;
    }

    /// <summary>Rewrites a raw Tencent hyvideo key (Kijai repacks: <c>img_attn_qkv</c>, <c>mlp.fc1</c>,
    /// <c>mod.linear</c>, <c>t_embedder.mlp.0</c>, <c>c_embedder.linear_1</c>) to the Comfy Flux-style name
    /// <see cref="MapOriginal"/> expects. Order matters: embedder <c>.mlp.0/.2</c> renames run BEFORE the refiner
    /// <c>mlp.fc1/fc2 → mlp.0/mlp.2</c> renames so the two never collide.</summary>
    private static string NormalizeTencentRaw(string key) => key
        .Replace(".mlp.0.", ".in_layer.")                      // time_in/guidance_in/t_embedder MLPs
        .Replace(".mlp.2.", ".out_layer.")
        .Replace("c_embedder.linear_1.", "c_embedder.in_layer.")
        .Replace("c_embedder.linear_2.", "c_embedder.out_layer.")
        .Replace("_attn_qkv.", "_attn.qkv.")
        .Replace("_attn_proj.", "_attn.proj.")
        .Replace("_attn_q_norm.", "_attn.norm.query_norm.")
        .Replace("_attn_k_norm.", "_attn.norm.key_norm.")
        .Replace(".q_norm.", ".norm.query_norm.")              // single_blocks (attn norms are underscore-joined)
        .Replace(".k_norm.", ".norm.key_norm.")
        .Replace("_mlp.fc1.", "_mlp.0.")                       // double-block img/txt MLPs
        .Replace("_mlp.fc2.", "_mlp.2.")
        .Replace(".mlp.fc1.", ".mlp.0.")                       // refiner-block MLPs
        .Replace(".mlp.fc2.", ".mlp.2.")
        .Replace("_mod.linear.", "_mod.lin.")
        .Replace(".modulation.linear.", ".modulation.lin.")
        .Replace("self_attn_qkv.", "self_attn.qkv.")
        .Replace("self_attn_proj.", "self_attn.proj.");

    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"HunyuanVideo DiT checkpoint not found: {checkpointPath}");
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        return (Convert(loader.GetAllTensors()), loader);
    }

    // ── Original Tencent/Comfy (Flux-style) naming ────────────────────────────────────────────

    private static void MapOriginal(string key, Tensor t, Dictionary<string, Tensor> o)
    {
        // Head.
        if (key == "img_in.proj.weight") { o["img_in.weight"] = Reshape2D(t, (int)t.Shape[0]); return; }
        if (key == "img_in.proj.bias") { o["img_in.bias"] = t; return; }
        if (TryMlpEmbedder(key, t, "time_in", "time_in", o)) return;
        if (TryMlpEmbedder(key, t, "vector_in", "vector_in", o)) return;
        if (TryMlpEmbedder(key, t, "guidance_in", "guidance_in", o)) return;
        if (key.StartsWith("txt_in.", StringComparison.Ordinal)) { o[key] = t; return; }   // refiner: 1:1
        if (key == "final_layer.linear.weight") { o["final_layer.proj.weight"] = t; return; }
        if (key == "final_layer.linear.bias") { o["final_layer.proj.bias"] = t; return; }
        if (key == "final_layer.adaLN_modulation.1.weight") { o["final_layer.mod.weight"] = t; return; }
        if (key == "final_layer.adaLN_modulation.1.bias") { o["final_layer.mod.bias"] = t; return; }

        if (key.StartsWith("double_blocks.", StringComparison.Ordinal)) { MapDoubleOriginal(key, t, o); return; }
        if (key.StartsWith("single_blocks.", StringComparison.Ordinal)) { MapSingleOriginal(key, t, o); return; }
        // Unknown / unused (e.g. byt5, vision) — drop silently.
    }

    private static void MapDoubleOriginal(string key, Tensor t, Dictionary<string, Tensor> o)
    {
        // key = double_blocks.{i}.<rest>
        int dot = key.IndexOf('.', "double_blocks.".Length);
        string i = key["double_blocks.".Length..dot];
        string rest = key[(dot + 1)..];
        string pre = $"double_blocks.{i}.";

        switch (rest)
        {
            case "img_mod.lin.weight": o[pre + "norm1.linear.weight"] = t; return;
            case "img_mod.lin.bias": o[pre + "norm1.linear.bias"] = t; return;
            case "txt_mod.lin.weight": o[pre + "norm1_context.linear.weight"] = t; return;
            case "txt_mod.lin.bias": o[pre + "norm1_context.linear.bias"] = t; return;
            case "img_attn.qkv.weight": SplitQkv(t, pre, "attn.to_q", "attn.to_k", "attn.to_v", "weight", o); return;
            case "img_attn.qkv.bias": SplitQkv(t, pre, "attn.to_q", "attn.to_k", "attn.to_v", "bias", o); return;
            case "txt_attn.qkv.weight": SplitQkv(t, pre, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "weight", o); return;
            case "txt_attn.qkv.bias": SplitQkv(t, pre, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "bias", o); return;
            case "img_attn.norm.query_norm.weight": case "img_attn.norm.query_norm.scale": o[pre + "attn.norm_q.weight"] = t; return;
            case "img_attn.norm.key_norm.weight": case "img_attn.norm.key_norm.scale": o[pre + "attn.norm_k.weight"] = t; return;
            case "txt_attn.norm.query_norm.weight": case "txt_attn.norm.query_norm.scale": o[pre + "attn.norm_added_q.weight"] = t; return;
            case "txt_attn.norm.key_norm.weight": case "txt_attn.norm.key_norm.scale": o[pre + "attn.norm_added_k.weight"] = t; return;
            case "img_attn.proj.weight": o[pre + "attn.to_out.0.weight"] = t; return;
            case "img_attn.proj.bias": o[pre + "attn.to_out.0.bias"] = t; return;
            case "txt_attn.proj.weight": o[pre + "attn.to_add_out.weight"] = t; return;
            case "txt_attn.proj.bias": o[pre + "attn.to_add_out.bias"] = t; return;
            case "img_mlp.0.weight": o[pre + "ff.net.0.proj.weight"] = t; return;
            case "img_mlp.0.bias": o[pre + "ff.net.0.proj.bias"] = t; return;
            case "img_mlp.2.weight": o[pre + "ff.net.2.weight"] = t; return;
            case "img_mlp.2.bias": o[pre + "ff.net.2.bias"] = t; return;
            case "txt_mlp.0.weight": o[pre + "ff_context.net.0.proj.weight"] = t; return;
            case "txt_mlp.0.bias": o[pre + "ff_context.net.0.proj.bias"] = t; return;
            case "txt_mlp.2.weight": o[pre + "ff_context.net.2.weight"] = t; return;
            case "txt_mlp.2.bias": o[pre + "ff_context.net.2.bias"] = t; return;
        }
    }

    private static void MapSingleOriginal(string key, Tensor t, Dictionary<string, Tensor> o)
    {
        int dot = key.IndexOf('.', "single_blocks.".Length);
        string i = key["single_blocks.".Length..dot];
        string rest = key[(dot + 1)..];
        string pre = $"single_blocks.{i}.";

        switch (rest)
        {
            case "modulation.lin.weight": o[pre + "norm.linear.weight"] = t; return;
            case "modulation.lin.bias": o[pre + "norm.linear.bias"] = t; return;
            case "norm.query_norm.weight": case "norm.query_norm.scale": o[pre + "attn.norm_q.weight"] = t; return;
            case "norm.key_norm.weight": case "norm.key_norm.scale": o[pre + "attn.norm_k.weight"] = t; return;
            case "linear2.weight": o[pre + "proj_out.weight"] = t; return;
            case "linear2.bias": o[pre + "proj_out.bias"] = t; return;
            case "linear1.weight": SplitLinear1(t, pre, o); return;
            case "linear1.bias": SplitLinear1(t, pre, o); return;
        }
    }

    // ── diffusers folder naming ───────────────────────────────────────────────────────────────

    private static void MapDiffusers(string key, Tensor t, Dictionary<string, Tensor> o)
    {
        if (key == "x_embedder.proj.weight") { o["img_in.weight"] = Reshape2D(t, (int)t.Shape[0]); return; }
        if (key == "x_embedder.proj.bias") { o["img_in.bias"] = t; return; }
        if (Rename(key, "time_text_embed.timestep_embedder.linear_1", "time_in.0", t, o)) return;
        if (Rename(key, "time_text_embed.timestep_embedder.linear_2", "time_in.2", t, o)) return;
        if (Rename(key, "time_text_embed.guidance_embedder.linear_1", "guidance_in.0", t, o)) return;
        if (Rename(key, "time_text_embed.guidance_embedder.linear_2", "guidance_in.2", t, o)) return;
        if (Rename(key, "time_text_embed.text_embedder.linear_1", "vector_in.0", t, o)) return;
        if (Rename(key, "time_text_embed.text_embedder.linear_2", "vector_in.2", t, o)) return;
        if (key.StartsWith("context_embedder.", StringComparison.Ordinal)) { MapRefinerDiffusers(key, t, o); return; }
        if (key == "norm_out.linear.weight") { o["final_layer.mod.weight"] = SwapHalves(t, (int)t.Shape[0] / 2); return; }
        if (key == "norm_out.linear.bias") { o["final_layer.mod.bias"] = SwapHalves(t, (int)t.Shape[0] / 2); return; }
        if (key == "proj_out.weight") { o["final_layer.proj.weight"] = t; return; }
        if (key == "proj_out.bias") { o["final_layer.proj.bias"] = t; return; }
        if (key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
        { o["double_blocks." + key["transformer_blocks.".Length..]] = t; return; }        // sub-keys already match
        if (key.StartsWith("single_transformer_blocks.", StringComparison.Ordinal))
        { o["single_blocks." + key["single_transformer_blocks.".Length..]] = t; return; }
    }

    private static void MapRefinerDiffusers(string key, Tensor t, Dictionary<string, Tensor> o)
    {
        string r = key["context_embedder.".Length..];
        if (Rename(r, "proj_in", "txt_in.input_embedder", t, o)) return;
        if (Rename(r, "time_text_embed.timestep_embedder.linear_1", "txt_in.t_embedder.in_layer", t, o)) return;
        if (Rename(r, "time_text_embed.timestep_embedder.linear_2", "txt_in.t_embedder.out_layer", t, o)) return;
        if (Rename(r, "time_text_embed.text_embedder.linear_1", "txt_in.c_embedder.in_layer", t, o)) return;
        if (Rename(r, "time_text_embed.text_embedder.linear_2", "txt_in.c_embedder.out_layer", t, o)) return;
        // token_refiner.refiner_blocks.{i}.* → txt_in.individual_token_refiner.blocks.{i}.*
        const string rb = "token_refiner.refiner_blocks.";
        if (r.StartsWith(rb, StringComparison.Ordinal))
        {
            int dot = r.IndexOf('.', rb.Length);
            string i = r[rb.Length..dot];
            string rest = r[(dot + 1)..];
            string pre = $"txt_in.individual_token_refiner.blocks.{i}.";
            switch (rest)
            {
                case "norm1.weight": o[pre + "norm1.weight"] = t; return;
                case "norm1.bias": o[pre + "norm1.bias"] = t; return;
                case "norm2.weight": o[pre + "norm2.weight"] = t; return;
                case "norm2.bias": o[pre + "norm2.bias"] = t; return;
                case "attn.to_out.0.weight": o[pre + "self_attn.proj.weight"] = t; return;
                case "attn.to_out.0.bias": o[pre + "self_attn.proj.bias"] = t; return;
                case "ff.net.0.proj.weight": o[pre + "mlp.0.weight"] = t; return;
                case "ff.net.0.proj.bias": o[pre + "mlp.0.bias"] = t; return;
                case "ff.net.2.weight": o[pre + "mlp.2.weight"] = t; return;
                case "ff.net.2.bias": o[pre + "mlp.2.bias"] = t; return;
                case "norm_out.linear.weight": o[pre + "adaLN_modulation.1.weight"] = t; return;
                case "norm_out.linear.bias": o[pre + "adaLN_modulation.1.bias"] = t; return;
                // Split Q/K/V are re-fused into self_attn.qkv on the last of the trio to arrive; buffer via temp keys.
                case "attn.to_q.weight": case "attn.to_k.weight": case "attn.to_v.weight":
                case "attn.to_q.bias": case "attn.to_k.bias": case "attn.to_v.bias":
                    o[pre + "__fuse." + rest] = t; TryFuseRefinerQkv(pre, o); return;
            }
        }
    }

    /// <summary>Fuses buffered diffusers refiner <c>attn.to_q/to_k/to_v</c> into a single <c>self_attn.qkv</c> once all
    /// three (weight or bias) are present, then removes the temporaries.</summary>
    private static void TryFuseRefinerQkv(string pre, Dictionary<string, Tensor> o)
    {
        foreach (string suf in new[] { "weight", "bias" })
        {
            string kq = pre + "__fuse.attn.to_q." + suf, kk = pre + "__fuse.attn.to_k." + suf, kv = pre + "__fuse.attn.to_v." + suf;
            if (o.ContainsKey(kq) && o.ContainsKey(kk) && o.ContainsKey(kv))
            {
                o[pre + "self_attn.qkv." + suf] = ConcatRows(o[kq], o[kk], o[kv]);
                o.Remove(kq); o.Remove(kk); o.Remove(kv);
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────────────────────

    private static bool TryMlpEmbedder(string key, Tensor t, string srcMod, string dstMod, Dictionary<string, Tensor> o)
    {
        if (Rename(key, srcMod + ".in_layer", dstMod + ".0", t, o)) return true;
        if (Rename(key, srcMod + ".out_layer", dstMod + ".2", t, o)) return true;
        return false;
    }

    /// <summary>Maps <c>{src}.weight</c>/<c>{src}.bias</c> → <c>{dst}.weight</c>/<c>{dst}.bias</c>. Returns true on match.</summary>
    private static bool Rename(string key, string src, string dst, Tensor t, Dictionary<string, Tensor> o)
    {
        if (key == src + ".weight") { o[dst + ".weight"] = t; return true; }
        if (key == src + ".bias") { o[dst + ".bias"] = t; return true; }
        return false;
    }

    /// <summary>Splits a fused QKV tensor [3H, ...] into three [H, ...] row-blocks under <c>{pre}{qName}.{suf}</c> etc.</summary>
    private static void SplitQkv(Tensor fused, string pre, string qName, string kName, string vName, string suf, Dictionary<string, Tensor> o)
    {
        int total = (int)fused.Shape[0];
        int h = total / 3;
        o[pre + qName + "." + suf] = RowSlice(fused, 0, h);
        o[pre + kName + "." + suf] = RowSlice(fused, h, h);
        o[pre + vName + "." + suf] = RowSlice(fused, 2 * h, h);
    }

    /// <summary>Splits the single-block <c>linear1</c> ([3H + mlp, ...]) into Q/K/V ([H,…]) + <c>proj_mlp</c> ([mlp,…]).
    /// Applies to weight and bias by the tensor's dim-0.</summary>
    private static void SplitLinear1(Tensor fused, string pre, Dictionary<string, Tensor> o)
    {
        // hidden is inferred from the trailing (input) dim for weights, or must be derived from 3H+mlp for bias.
        // For HunyuanVideo hidden=3072, mlp=4*hidden. rows = 3*hidden + 4*hidden = 7*hidden → hidden = rows/7.
        int rows = (int)fused.Shape[0];
        int hidden = rows / 7;
        int mlp = rows - 3 * hidden;
        string suf = fused.Shape.Rank == 2 ? "weight" : "bias";
        o[pre + "attn.to_q." + suf] = RowSlice(fused, 0, hidden);
        o[pre + "attn.to_k." + suf] = RowSlice(fused, hidden, hidden);
        o[pre + "attn.to_v." + suf] = RowSlice(fused, 2 * hidden, hidden);
        o[pre + "proj_mlp." + suf] = RowSlice(fused, 3 * hidden, mlp);
    }

    /// <summary>Copies rows <c>[startRow, startRow+numRows)</c> of <paramref name="src"/> into a new tensor (same dtype).
    /// A "row" is the product of all dims after dim-0 (or 1 for a rank-1 bias).</summary>
    private static Tensor RowSlice(Tensor src, int startRow, int numRows)
    {
        long cols = src.Shape.Rank == 1 ? 1 : src.ElementCount / src.Shape[0];
        int esz = src.DType.SizeInBytes;
        long rowBytes = cols * esz;
        TensorShape shape = src.Shape.Rank == 1 ? new TensorShape(numRows) : new TensorShape(numRows, cols);
        Tensor dst = new(shape, src.DType);
        // Propagate fp8_scaled per-tensor scale — the raw fp8 bytes are real_value/scale, so a slice must keep the source's factor.
        dst.Fp8ScaleFactor = src.Fp8ScaleFactor;
        byte* sp = (byte*)src.DataPointer + startRow * rowBytes;
        Buffer.MemoryCopy(sp, (void*)dst.DataPointer, numRows * rowBytes, numRows * rowBytes);
        return dst;
    }

    /// <summary>Concatenates three row-blocks (Q, K, V) along dim-0 into one tensor (same dtype).</summary>
    private static Tensor ConcatRows(Tensor q, Tensor k, Tensor v)
    {
        long cols = q.Shape.Rank == 1 ? 1 : q.ElementCount / q.Shape[0];
        int esz = q.DType.SizeInBytes;
        int rows = (int)(q.Shape[0] + k.Shape[0] + v.Shape[0]);
        TensorShape shape = q.Shape.Rank == 1 ? new TensorShape(rows) : new TensorShape(rows, cols);
        Tensor dst = new(shape, q.DType);
        // Propagate fp8_scaled per-tensor scale (q's — a fused tensor carries one factor; diffusers refiner q/k/v ship unscaled or with identical scales in practice).
        dst.Fp8ScaleFactor = q.Fp8ScaleFactor;
        byte* dp = (byte*)dst.DataPointer;
        long qb = q.ElementCount * esz, kb = k.ElementCount * esz, vb = v.ElementCount * esz;
        Buffer.MemoryCopy((void*)q.DataPointer, dp, qb, qb);
        Buffer.MemoryCopy((void*)k.DataPointer, dp + qb, kb, kb);
        Buffer.MemoryCopy((void*)v.DataPointer, dp + qb + kb, vb, vb);
        return dst;
    }

    /// <summary>Reshapes a Conv3d patch-embed weight [out, in, kt, kh, kw] to a 2D Linear weight [out, in·kt·kh·kw]
    /// (identity byte layout; the flattened (in, kt, kh, kw) order matches the DiT's patchify vector order).</summary>
    private static Tensor Reshape2D(Tensor src, int rows)
    {
        long cols = src.ElementCount / rows;
        Tensor dst = new(new TensorShape(rows, cols), src.DType);
        // Propagate fp8_scaled per-tensor scale — the reshape is byte-identical, so the source's factor still applies.
        dst.Fp8ScaleFactor = src.Fp8ScaleFactor;
        long bytes = src.ElementCount * src.DType.SizeInBytes;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
        return dst;
    }

    /// <summary>Swaps the two halves along dim-0 of a modulation tensor: diffusers <c>norm_out</c> chunks
    /// <c>[scale, shift]</c>; the DiT's final <c>Modulate</c> reads <c>[shift, scale]</c>. F32 only (dequant runs first).</summary>
    private static Tensor SwapHalves(Tensor src, int half)
    {
        Tensor f = src.DType == DType.F32 ? src : src.CastTo(DType.F32);
        long cols = f.Shape.Rank == 1 ? 1 : f.ElementCount / f.Shape[0];
        Tensor dst = new(f.Shape, DType.F32);
        float* sp = (float*)f.DataPointer; float* dp = (float*)dst.DataPointer;
        long block = half * cols;
        Buffer.MemoryCopy(sp + block, dp, block * 4, block * 4);   // src second half (shift) → dst first half
        Buffer.MemoryCopy(sp, dp + block, block * 4, block * 4);   // src first half  (scale) → dst second half
        return dst;
    }

    /// <summary>Converts a HunyuanVideo VAE checkpoint's DECODER from ldm/Tencent naming to the diffusers naming
    /// <see cref="HartsyInference.Diffusion.Models.Vae.HunyuanVideoVaeDecoder"/> expects: <c>decoder.up.N</c>
    /// (reversed) → <c>decoder.up_blocks.(numUp−1−N)</c>, <c>mid.block_1/2</c> → <c>mid_block.resnets.0/1</c>,
    /// <c>mid.attn_1.{norm,q,k,v,proj_out}</c> → <c>mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}</c>
    /// (attn projections reshaped <c>[C,C,1,1,1]→[C,C]</c>), <c>nin_shortcut</c> → <c>conv_shortcut</c>,
    /// <c>norm_out</c> → <c>conv_norm_out</c>. Keeps <c>post_quant_conv</c>; drops the encoder + <c>quant_conv</c>
    /// (decode-only). fp8 dequant runs first.</summary>
    public static Dictionary<string, Tensor> ConvertVaeDecoder(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);
        int numUp = 0;
        foreach (string k in allWeights.Keys)
            if (k.StartsWith("decoder.up.", StringComparison.Ordinal))
            {
                int dot = k.IndexOf('.', "decoder.up.".Length);
                if (dot > 0 && int.TryParse(k["decoder.up.".Length..dot], out int n)) numUp = Math.Max(numUp, n + 1);
            }

        Dictionary<string, Tensor> o = new(allWeights.Count);
        foreach ((string key, Tensor t) in allWeights)
        {
            if (key is "post_quant_conv.weight" or "post_quant_conv.bias") { o[key] = t; continue; }
            if (!key.StartsWith("decoder.", StringComparison.Ordinal)) continue;   // drop encoder / quant_conv
            string r = key["decoder.".Length..];
            if (r.StartsWith("conv_in.", StringComparison.Ordinal) || r.StartsWith("conv_out.", StringComparison.Ordinal)) { o[key] = t; continue; }
            if (r.StartsWith("norm_out.", StringComparison.Ordinal)) { o["decoder.conv_norm_out." + r["norm_out.".Length..]] = t; continue; }
            if (r.StartsWith("mid.block_1.", StringComparison.Ordinal)) { o["decoder.mid_block.resnets.0." + r["mid.block_1.".Length..]] = t; continue; }
            if (r.StartsWith("mid.block_2.", StringComparison.Ordinal)) { o["decoder.mid_block.resnets.1." + r["mid.block_2.".Length..]] = t; continue; }
            if (r.StartsWith("mid.attn_1.", StringComparison.Ordinal))
            {
                string s = r["mid.attn_1.".Length..];
                const string ab = "decoder.mid_block.attentions.0.";
                if (s.StartsWith("norm.", StringComparison.Ordinal)) o[ab + "group_norm." + s["norm.".Length..]] = t;
                else if (s.StartsWith("q.", StringComparison.Ordinal)) o[ab + "to_q." + s["q.".Length..]] = AttnProj(t, s);
                else if (s.StartsWith("k.", StringComparison.Ordinal)) o[ab + "to_k." + s["k.".Length..]] = AttnProj(t, s);
                else if (s.StartsWith("v.", StringComparison.Ordinal)) o[ab + "to_v." + s["v.".Length..]] = AttnProj(t, s);
                else if (s.StartsWith("proj_out.", StringComparison.Ordinal)) o[ab + "to_out.0." + s["proj_out.".Length..]] = AttnProj(t, s);
                continue;
            }
            if (r.StartsWith("up.", StringComparison.Ordinal))
            {
                int dot = r.IndexOf('.', 3);
                int n = int.Parse(r[3..dot]);
                int di = numUp - 1 - n;
                string rest = r[(dot + 1)..];
                if (rest.StartsWith("block.", StringComparison.Ordinal))
                {
                    string bm = rest["block.".Length..];
                    int d2 = bm.IndexOf('.');
                    int m = int.Parse(bm[..d2]);
                    string sub = bm[(d2 + 1)..];
                    string tgt = sub.StartsWith("nin_shortcut.", StringComparison.Ordinal) ? "conv_shortcut." + sub["nin_shortcut.".Length..] : sub;
                    o[$"decoder.up_blocks.{di}.resnets.{m}.{tgt}"] = t;
                }
                else if (rest.StartsWith("upsample.conv.", StringComparison.Ordinal))
                    o[$"decoder.up_blocks.{di}.upsamplers.0.conv." + rest["upsample.conv.".Length..]] = t;
                continue;
            }
        }
        return o;

        // ldm attention q/k/v/proj_out are 1×1×1 convs [C,C,1,1,1]; the diffusers VaeAttention uses Linear [C,C].
        static Tensor AttnProj(Tensor t, string sub) =>
            sub.EndsWith(".weight", StringComparison.Ordinal) && t.Shape.Rank > 2 ? Reshape2D(t, (int)t.Shape[0]) : t;
    }
}
