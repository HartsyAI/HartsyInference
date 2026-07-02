using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

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
}
