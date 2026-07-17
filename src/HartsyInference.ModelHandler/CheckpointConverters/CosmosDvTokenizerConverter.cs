using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads the weights of the Cosmos-Tokenize1-DV8x16x16-720p encoder/decoder into a plain
/// <c>Dictionary&lt;string, Tensor&gt;</c> for <c>HartsyInference.Video.Tokenizers.CosmosDvTokenizer</c> to build its
/// causal-conv stacks from. NVIDIA ships the tokenizer only as TorchScript (<c>encoder.jit</c> / <c>decoder.jit</c> /
/// <c>ema.jit</c>) — no safetensors (research-doc Open Q §10). This converter consumes the weights <b>after</b> a
/// one-off off-ship JIT→safetensors extraction (the same "convert once" posture as the AR <c>model.pt</c>); it does
/// not itself introspect TorchScript.
///
/// <para>Returns raw prefix-stripped weights. The exact causal-conv stage schedule (channels, resnet/attention
/// blocks) — needed to instantiate the <c>CausalConv3d</c> stacks — is resolved in the Video package at integration
/// from these tensors' shapes; this stays in ModelHandler because it cannot reference Diffusion's conv types
/// (that would invert the dependency edge).</para></summary>
public static class CosmosDvTokenizerConverter
{
    private static readonly string[] StripPrefixes = ["module.", "model.", "net.", "_orig_mod."];

    /// <summary>Loads an extracted encoder or decoder weight file (safetensors, or a <c>.pt</c> pickle) into a
    /// prefix-stripped weight dict plus its backing loader.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) Load(string path, bool castToF32 = false)
    {
        (Dictionary<string, Tensor> raw, IDisposable loader) = ReadRaw(path);
        Dictionary<string, Tensor> mapped = new(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = StripPrefix(kv.Key);
            Tensor value = kv.Value;
            if (castToF32 && value.DType != DType.F32 && !value.DType.IsQuantized)
                value = value.CastTo(DType.F32);
            mapped[key] = value;
        }
        return (mapped, loader);
    }

    private static string StripPrefix(string key)
    {
        foreach (string p in StripPrefixes)
            if (key.StartsWith(p, StringComparison.Ordinal))
                return key.Substring(p.Length);
        return key;
    }

    private static (Dictionary<string, Tensor> Raw, IDisposable Loader) ReadRaw(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader st = new();
            st.Load(path);
            return (st.GetAllTensors(), st);
        }
        PytorchPickleLoader pt = new();
        pt.Load(path);
        return (pt.GetAllTensors(), pt);
    }
}
