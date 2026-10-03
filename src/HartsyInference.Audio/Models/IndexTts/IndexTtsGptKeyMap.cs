using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>Translates IndexTTS's real <c>model.gpt.h.{i}.*</c> checkpoint keys (a standard HuggingFace <c>GPT2Model</c>
/// export) into the key scheme <see cref="HartsyInference.Audio.Models.LanguageModels.Gpt.GptBlock"/> expects.</summary>
/// <remarks>Two differences from the Bark checkpoints <c>GptBlock</c> was built against, both verified against the
/// real <c>gpt.pth</c> state dict: (1) HF's <c>Conv1D</c> layer (used for <c>c_attn</c>/<c>c_proj</c>/<c>c_fc</c>)
/// stores its weight as <c>[inFeatures, outFeatures]</c> — the transpose of <c>nn.Linear</c>'s <c>[out, in]</c>
/// that <c>GptBlock</c>/<see cref="HartsyInference.Audio.Models.Whisper.WhisperOps.ProjectLinear"/> assume — so
/// every projection weight needs transposing at load time, not just renaming; (2) biases are present throughout
/// (confirmed real, non-empty bias tensors in the checkpoint), unlike Bark's bias-free convention.</remarks>
internal static unsafe class IndexTtsGptKeyMap
{
    /// <summary>Returns a new dictionary containing <paramref name="raw"/>'s entries plus a <c>{blockPrefix}.{i}.*</c>
    /// entry per layer under <see cref="HartsyInference.Audio.Models.LanguageModels.Gpt.GptBlock"/>'s expected
    /// sub-keys, so the result can be passed directly to <c>GptBackbone.LoadWeights</c>, plus the freshly allocated
    /// transposed weight tensors the caller must dispose. <see cref="HartsyInference.Audio.Models.LanguageModels.Gpt.GptBlock.Dispose"/>
    /// is a no-op by design — Bark's weights are all borrowed references the checkpoint loader owns — but
    /// <see cref="TransposeMatrix"/> allocates genuinely new tensors nothing else references, so leaving disposal
    /// to <see cref="HartsyInference.Audio.Models.LanguageModels.Gpt.GptBlock"/> would leak them.</summary>
    public static (Dictionary<string, Tensor> Weights, Tensor[] OwnedTensors) Translate(
        IReadOnlyDictionary<string, Tensor> raw, string rawPrefix, string blockPrefix, int numLayers)
    {
        Dictionary<string, Tensor> w = new(raw);
        Tensor[] owned = new Tensor[numLayers * 4];
        int oi = 0;
        try
        {
            for (int i = 0; i < numLayers; i++)
            {
                string src = $"{rawPrefix}.{i}";
                string dst = $"{blockPrefix}.{i}";
                w[$"{dst}.layernorm_1.weight"] = raw[$"{src}.ln_1.weight"];
                w[$"{dst}.layernorm_1.bias"] = raw[$"{src}.ln_1.bias"];
                Tensor attProjW = TransposeMatrix(raw[$"{src}.attn.c_attn.weight"]);
                w[$"{dst}.attn.att_proj.weight"] = owned[oi++] = attProjW;
                w[$"{dst}.attn.att_proj.bias"] = raw[$"{src}.attn.c_attn.bias"];
                Tensor outProjW = TransposeMatrix(raw[$"{src}.attn.c_proj.weight"]);
                w[$"{dst}.attn.out_proj.weight"] = owned[oi++] = outProjW;
                w[$"{dst}.attn.out_proj.bias"] = raw[$"{src}.attn.c_proj.bias"];
                w[$"{dst}.layernorm_2.weight"] = raw[$"{src}.ln_2.weight"];
                w[$"{dst}.layernorm_2.bias"] = raw[$"{src}.ln_2.bias"];
                Tensor mlpInW = TransposeMatrix(raw[$"{src}.mlp.c_fc.weight"]);
                w[$"{dst}.mlp.in_proj.weight"] = owned[oi++] = mlpInW;
                w[$"{dst}.mlp.in_proj.bias"] = raw[$"{src}.mlp.c_fc.bias"];
                Tensor mlpOutW = TransposeMatrix(raw[$"{src}.mlp.c_proj.weight"]);
                w[$"{dst}.mlp.out_proj.weight"] = owned[oi++] = mlpOutW;
                w[$"{dst}.mlp.out_proj.bias"] = raw[$"{src}.mlp.c_proj.bias"];
            }
        }
        catch
        {
            // A truncated/incompatible checkpoint can throw partway through (e.g. a missing key on layer 5 of
            // 24) — every transpose already completed for earlier layers is a real tensor nothing else
            // references yet (the caller never receives `owned` to dispose them), so they must be freed here.
            for (int j = 0; j < oi; j++) owned[j].Dispose();
            throw;
        }
        return (w, owned);
    }

    /// <summary>Transposes a 2-D <c>[rows, cols]</c> tensor to <c>[cols, rows]</c> (HF <c>Conv1D</c> → <c>nn.Linear</c> convention).</summary>
    private static Tensor TransposeMatrix(Tensor src)
    {
        Tensor f32 = src.DType == DType.F32 ? src : src.CastTo(DType.F32);
        int rows = (int)f32.Shape[0], cols = (int)f32.Shape[1];
        Tensor dst = new(new TensorShape(cols, rows), DType.F32);
        float* sp = (float*)f32.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            float* srow = sp + (long)r * cols;
            for (int c = 0; c < cols; c++) dp[(long)c * rows + r] = srow[c];
        }
        if (!ReferenceEquals(f32, src)) f32.Dispose();
        return dst;
    }
}
