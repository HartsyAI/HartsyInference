using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>The ComfyUI attention hints in the Fibo 2.0 Comfy-Org int8 builds (see <see cref="ComfyAttentionHint"/>). The engine runs
/// float attention by default; INT8 attention is an opt-in knob (<c>numerics.int8AttnV2</c>, CUDA), numerically unverified against
/// ComfyUI. A malformed hint is refused; a hint naming a backend is accepted with a warning (see <see cref="Describe"/>).</summary>
internal static class Fibo2AttentionHints
{
    /// <summary>ComfyUI's int8 attention backend, the one the Fibo 2.0 and RMBG 3.0 int8 builds name.</summary>
    public const string ComfyKitchenInt8 = "comfy_kitchen_int8";

    /// <summary>Validates every hint in <paramref name="weights"/> and returns the distinct backend names they name, in file order.
    /// Hints are not weights; the caller drops them.</summary>
    /// <exception cref="InvalidDataException">A hint is not a U8 blob holding <c>{"attention": "&lt;backend&gt;"}</c>.</exception>
    public static IReadOnlyList<string> Requested(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        List<string> names = [];
        foreach (KeyValuePair<string, Tensor> pair in weights)
        {
            if (!ComfyAttentionHint.IsHint(pair.Key))
                continue;
            string backend = Read(pair.Value, pair.Key);
            if (!names.Contains(backend))
                names.Add(backend);
        }
        return names;
    }

    /// <summary>The warning logged for a backend the checkpoint names. The engine runs float attention by default for all of them.</summary>
    public static string Describe(string backend)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(backend);
        return backend == ComfyKitchenInt8
            ? $"[Fibo2Recipe] The checkpoint names {ComfyKitchenInt8} attention. The engine runs float attention by default; INT8 "
                + "attention is an opt-in knob (numerics.int8AttnV2), numerically unverified against Comfy."
            : $"[Fibo2Recipe] The checkpoint names attention backend '{backend}'. The engine runs float attention by default; outputs "
                + "may differ from ComfyUI.";
    }

    /// <summary>Decodes one hint blob to its backend name.</summary>
    internal static string Read(Tensor blob, string key)
    {
        ArgumentNullException.ThrowIfNull(blob);
        if (blob.DType != DType.U8 || blob.Shape.Rank != 1 || blob.ElementCount is <= 0 or > 4096)
        {
            throw new InvalidDataException($"Fibo 2.0 tensor '{key}' must be a one-dimensional U8 blob of 1 to 4096 bytes; got {blob.DType} {blob.Shape}.");
        }
        string? backend = ComfyAttentionHint.TryParseBackend(blob.AsReadOnlySpan<byte>());
        return backend ?? throw new InvalidDataException(
            $"Fibo 2.0 tensor '{key}' is not an attention hint of the form {{\"attention\": \"<backend>\"}}.");
    }
}
