using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Csm;

/// <summary>Remaps <c>unsloth/csm-1b</c>'s raw key layout (a non-gated mirror of the HF
/// <c>transformers</c> <c>CsmForConditionalGeneration</c> export) onto the flat key names
/// <see cref="CsmModel.LoadWeights"/> expects.
///
/// <para>The <c>transformers</c> export nests the backbone/depth-decoder under
/// <c>backbone_model.</c>/<c>depth_decoder.model.</c> prefixes (otherwise identical to the
/// engine's standard Llama key names — <c>self_attn.q_proj</c>, <c>input_layernorm.weight</c>,
/// <c>mlp.gate_proj/up_proj/down_proj</c> — so <see cref="Qwen2Model.LoadWeightsHeadless"/> reads
/// them unmodified once re-prefixed), and stores the audio embedding table + codebook heads as two
/// combined tensors rather than the per-codebook slices <see cref="CsmModel.LoadWeights"/> reads.
/// Mirrors <c>tests/python-reference/csm_reference/dump_csm_reference.py</c>'s conversion exactly.</para>
///
/// <para>An earlier non-gated mirror choice, <c>nielsr/csm-1b</c>, ships the ORIGINAL torchtune-style
/// keys instead (<c>attn.q_proj</c>, <c>sa_norm.scale</c>, <c>mlp.w1/w2/w3</c>) — confirmed by reading
/// its safetensors header directly, which is the KeyNotFoundException root cause. <c>unsloth/csm-1b</c>
/// is also ungated and already carries the matching key convention, so the fix is switching mirrors
/// (see <see cref="HartsyInference.Engine.Audio.Tts.TtsCatalog.Csm"/>) plus this remap for the two
/// combined tensors.</para></summary>
public static class CsmWeightRemap
{
    private const int NumCodebooks = 32;
    private const int AudioVocab = 2_051;

    /// <summary>Converts a raw <c>unsloth/csm-1b</c> weight dict into the key layout
    /// <see cref="CsmModel.LoadWeights"/> reads. Returns a new dictionary; tensors that only need
    /// renaming are re-keyed (same instance), the two combined tensors are split into fresh owned
    /// per-codebook tensors.</summary>
    public static unsafe Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> w = new(raw.Count + NumCodebooks * 2, StringComparer.Ordinal);

        foreach ((string key, Tensor t) in raw)
        {
            if (key.StartsWith("codec_model.", StringComparison.Ordinal))
            {
                continue;   // Mimi codec weights are baked into this checkpoint too; loaded separately from kyutai/mimi.
            }
            if (key.StartsWith("backbone_model.", StringComparison.Ordinal))
            {
                w["backbone." + key["backbone_model.".Length..]] = t;
            }
            else if (key.StartsWith("depth_decoder.model.layers.", StringComparison.Ordinal)
                || key == "depth_decoder.model.norm.weight")
            {
                w["decoder." + key["depth_decoder.model.".Length..]] = t;
            }
            else if (key == "embed_text_tokens.weight")
            {
                w["text_embeddings.weight"] = t;
            }
            else if (key == "lm_head.weight")
            {
                w["codebook0_head.weight"] = t;
            }
            else if (key == "depth_decoder.model.inputs_embeds_projector.weight")
            {
                w["projection.weight"] = t;
            }
            else if (key == "depth_decoder.model.embed_tokens.weight")
            {
                // [NumCodebooks * AudioVocab, hidden] -> NumCodebooks x [AudioVocab, hidden] row-block copies.
                // Codebook i's codes occupy rows [i*AudioVocab, (i+1)*AudioVocab) (see BuildDecoderSequence's
                // "code + j*vocab" indexing, which this layout was designed to match).
                int hidden = (int)t.Shape[1];
                long rowBytes = hidden * t.DType.SizeInBytes;
                for (int i = 0; i < NumCodebooks; i++)
                {
                    Tensor slice = new(new TensorShape(AudioVocab, hidden), t.DType);
                    Buffer.MemoryCopy(
                        (byte*)t.DataPointer + (long)i * AudioVocab * rowBytes,
                        (void*)slice.DataPointer,
                        AudioVocab * rowBytes, AudioVocab * rowBytes);
                    w[$"audio_embeddings.{i}.weight"] = slice;
                }
            }
            else if (key == "depth_decoder.codebooks_head.weight")
            {
                // [NumCodebooks-1, decoderHidden, AudioVocab] (out=decoderHidden, in=AudioVocab convention used by
                // the reference's einsum) -> per-codebook TRANSPOSE to [AudioVocab, decoderHidden] so it matches
                // every other head/projection in this codebase ([out, in], consumed by WhisperOps.ProjectLinear).
                int numHeads = (int)t.Shape[0];
                int decHidden = (int)t.Shape[1];
                Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
                float* src = (float*)f32.DataPointer;
                for (int i = 0; i < numHeads; i++)
                {
                    Tensor head = new(new TensorShape(AudioVocab, decHidden), DType.F32);
                    float* dst = (float*)head.DataPointer;
                    float* srcHead = src + (long)i * decHidden * AudioVocab;
                    for (int r = 0; r < decHidden; r++)
                        for (int c = 0; c < AudioVocab; c++)
                            dst[(long)c * decHidden + r] = srcHead[(long)r * AudioVocab + c];
                    w[$"audio_head.{i}.weight"] = head;
                }
                if (!ReferenceEquals(f32, t)) f32.Dispose();
            }
            else
            {
                w[key] = t;
            }
        }

        return w;
    }

    /// <summary>Extracts the Mimi codec bundled in the same checkpoint under the <c>codec_model.</c> prefix
    /// (32 codebooks: 1 semantic + 31 acoustic — verified against the safetensors header, matching
    /// <see cref="HartsyInference.Audio.Models.Codecs.Mimi.MimiConfig.Mimi24kHzDsm"/>). Already plain HF
    /// <c>transformers</c> <c>MimiModel</c> key layout (not the moshi-native DSM layout
    /// <see cref="HartsyInference.Audio.Models.Codecs.Mimi.MimiDsmWeights"/> adapts) — stripping the prefix is
    /// enough for <see cref="HartsyInference.Audio.Models.Codecs.Mimi.Mimi.LoadWeights"/> to read it directly.
    /// Using this bundled codec (instead of the separately-published 8-codebook <c>kyutai/mimi</c>) avoids a
    /// codebook-count mismatch: CSM's depth decoder predicts all 32 codebooks (see <see cref="CsmConfig.NumCodebooks"/>),
    /// which the standard 8-codebook Mimi can't consume.</summary>
    public static Dictionary<string, Tensor> ExtractMimiWeights(IReadOnlyDictionary<string, Tensor> raw)
    {
        const string prefix = "codec_model.";
        Dictionary<string, Tensor> w = new(raw.Count, StringComparer.Ordinal);
        foreach ((string key, Tensor t) in raw)
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                w[key[prefix.Length..]] = t;
        return w;
    }
}
