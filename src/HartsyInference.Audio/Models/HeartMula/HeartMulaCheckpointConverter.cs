using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>Remaps the published HeartMuLa-oss-3B checkpoint (torchtune <c>llama3_2</c> layout) onto the key
/// names the engine's <see cref="Csm.CsmModel"/> / <c>GenericTransformer</c> expect (HF-Llama names), and
/// splits the two fused tensors the upstream model stores as single parameters.
///
/// <para>Verified against the upstream <c>heartlib</c> (<c>modeling_heartmula.py</c>). Real → engine map:</para>
/// <list type="bullet">
/// <item><c>{bb}.layers.N.attn.q_proj/k_proj/v_proj</c> → <c>{bb}.layers.N.self_attn.q_proj/k_proj/v_proj</c></item>
/// <item><c>{bb}.layers.N.attn.output_proj</c> → <c>self_attn.o_proj</c></item>
/// <item><c>{bb}.layers.N.mlp.w1/w3/w2</c> → <c>mlp.gate_proj/up_proj/down_proj</c> (SwiGLU: w1=gate, w3=up, w2=down)</item>
/// <item><c>{bb}.layers.N.sa_norm.scale</c> → <c>input_layernorm.weight</c>; <c>mlp_norm.scale</c> → <c>post_attention_layernorm.weight</c></item>
/// <item><c>{bb}.norm.scale</c> → <c>{bb}.norm.weight</c> (bb ∈ {backbone, decoder})</item>
/// <item><c>audio_embeddings.weight</c> <c>[NumCodebooks·AudioVocab, H]</c> → split into
///   <c>audio_embeddings.{i}.weight</c> <c>[AudioVocab, H]</c> (row block <c>i·AudioVocab .. (i+1)·AudioVocab</c>,
///   matching <c>_embed_audio(i, t) = embed(t + i·audio_vocab)</c>).</item>
/// <item><c>audio_head</c> <c>[NumCodebooks-1, decoderH, AudioVocab]</c> (a single <c>nn.Parameter</c>, used as
///   <c>h @ audio_head[i]</c>, i.e. <c>[in, out]</c>) → split + transpose into <c>audio_head.{i}.weight</c>
///   <c>[AudioVocab, decoderH]</c> (the <c>[out, in]</c> layout <c>ProjectLinear</c> needs).</item>
/// </list>
/// <para><c>text_embeddings.weight</c>, <c>codebook0_head.weight</c>, <c>projection.weight</c>,
/// <c>muq_linear.{weight,bias}</c>, <c>unconditional_text_embedding.weight</c> pass through unchanged.</para></summary>
public static unsafe class HeartMulaCheckpointConverter
{
    /// <summary>Remaps a raw HeartMuLa checkpoint dictionary into the engine key layout. Pass-through tensors
    /// are shared (no copy); the two fused tensors are split into freshly-allocated F32 tensors.</summary>
    public static Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> w, HeartMulaConfig cfg)
    {
        int nb = cfg.Lm.NumCodebooks;
        int vocab = cfg.Lm.AudioVocab;
        int bbH = cfg.Lm.Backbone.HiddenSize;
        int decH = cfg.Lm.Decoder.HiddenSize;

        Dictionary<string, Tensor> outW = new(w.Count + 32);

        foreach ((string key, Tensor t) in w)
        {
            string? mapped = MapLayerKey(key);
            if (mapped is not null) { outW[mapped] = t; continue; }

            switch (key)
            {
                case "backbone.norm.scale": outW["backbone.norm.weight"] = t; break;
                case "decoder.norm.scale": outW["decoder.norm.weight"] = t; break;
                case "audio_embeddings.weight":
                    SplitAudioEmbeddings(outW, WhisperOps.EnsureF32(t), nb, vocab, bbH);
                    break;
                case "audio_head":
                    SplitAudioHead(outW, WhisperOps.EnsureF32(t), nb, vocab, decH);
                    break;
                default:
                    // text_embeddings / codebook0_head / projection / muq_linear / unconditional_text_embedding.
                    outW[key] = t;
                    break;
            }
        }
        return outW;
    }

    /// <summary>Maps a per-layer torchtune key onto its HF name, or null if it is not a layer weight.</summary>
    private static string? MapLayerKey(string key)
    {
        // Expect "<bb>.layers.<n>.<rest>"; bb ∈ {backbone, decoder}.
        int li = key.IndexOf(".layers.", StringComparison.Ordinal);
        if (li < 0) return null;
        int rest = key.IndexOf('.', li + ".layers.".Length);
        if (rest < 0) return null;
        string head = key[..rest];                  // "<bb>.layers.<n>"
        string tail = key[(rest + 1)..];            // e.g. "attn.q_proj.weight"

        string? newTail = tail switch
        {
            "attn.q_proj.weight" => "self_attn.q_proj.weight",
            "attn.k_proj.weight" => "self_attn.k_proj.weight",
            "attn.v_proj.weight" => "self_attn.v_proj.weight",
            "attn.output_proj.weight" => "self_attn.o_proj.weight",
            "mlp.w1.weight" => "mlp.gate_proj.weight",
            "mlp.w3.weight" => "mlp.up_proj.weight",
            "mlp.w2.weight" => "mlp.down_proj.weight",
            "sa_norm.scale" => "input_layernorm.weight",
            "mlp_norm.scale" => "post_attention_layernorm.weight",
            _ => null,
        };
        return newTail is null ? null : $"{head}.{newTail}";
    }

    /// <summary>Splits the fused <c>[nb·vocab, H]</c> table into <c>nb</c> contiguous <c>[vocab, H]</c> blocks.</summary>
    private static void SplitAudioEmbeddings(Dictionary<string, Tensor> outW, Tensor fused, int nb, int vocab, int h)
    {
        if (fused.Shape[0] != (long)nb * vocab || fused.Shape[1] != h)
            throw new InvalidOperationException($"audio_embeddings.weight is {fused.Shape}, expected [{(long)nb * vocab},{h}].");
        float* src = (float*)fused.DataPointer;
        for (int i = 0; i < nb; i++)
        {
            Tensor block = new(new TensorShape(vocab, h), DType.F32);
            long bytes = (long)vocab * h * sizeof(float);
            Buffer.MemoryCopy(src + (long)i * vocab * h, (void*)block.DataPointer, bytes, bytes);
            outW[$"audio_embeddings.{i}.weight"] = block;
        }
    }

    /// <summary>Splits + transposes the fused <c>audio_head</c> <c>[nb-1, decH, vocab]</c> Parameter into
    /// <c>nb-1</c> heads of <c>[vocab, decH]</c> (transpose the per-head <c>[decH, vocab]</c> = <c>[in, out]</c>
    /// slice so it matches <c>ProjectLinear</c>'s <c>[out, in]</c> weight layout).</summary>
    private static void SplitAudioHead(Dictionary<string, Tensor> outW, Tensor fused, int nb, int vocab, int decH)
    {
        if (fused.Shape[0] != nb - 1 || fused.Shape[1] != decH || fused.Shape[2] != vocab)
            throw new InvalidOperationException($"audio_head is {fused.Shape}, expected [{nb - 1},{decH},{vocab}].");
        float* src = (float*)fused.DataPointer;
        for (int i = 0; i < nb - 1; i++)
        {
            float* slice = src + (long)i * decH * vocab;   // [decH, vocab] = [in, out]
            Tensor head = new(new TensorShape(vocab, decH), DType.F32);   // [out, in]
            float* dst = (float*)head.DataPointer;
            for (int o = 0; o < vocab; o++)
                for (int inp = 0; inp < decH; inp++)
                    dst[(long)o * decH + inp] = slice[(long)inp * vocab + o];
            outW[$"audio_head.{i}.weight"] = head;
        }
    }
}
