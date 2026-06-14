using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf;

/// <summary>One-line bridge between <see cref="GgufModelLoader"/> and the existing per-model <c>*CheckpointConverter.Convert</c> methods. Every checkpoint converter in this codebase accepts a flat <c>Dictionary&lt;string, Tensor&gt;</c> and returns a <c>ConvertedWeights</c> record — that's exactly the shape <see cref="GgufModelLoader.LoadDequantized"/> produces. So the integration is one line per pipeline:
///
/// <code>
/// // Flux:
/// (var weights, var handle) = GgufConverterBridge.LoadGguf(ggufPath, DType.F16, FluxCheckpointConverter.Convert);
/// using (handle)
/// {
///     // weights is FluxCheckpointConverter.ConvertedWeights — same shape as the safetensors path.
///     // Pass to FluxTransformer.LoadWeights / ClipTextEncoder.LoadWeights / etc as usual.
/// }
/// </code>
///
/// <para>Same pattern for SDXL (<see cref="HartsyInference.ModelHandler.CheckpointConverters.SdxlCheckpointConverter.Convert"/>), SD3 (<see cref="HartsyInference.ModelHandler.CheckpointConverters.Sd3CheckpointConverter.Convert"/>), and every other model. **No per-pipeline shim is required** — the bridge is generic over the converter delegate.</para></summary>
public static class GgufConverterBridge
{
    /// <summary>Loads a GGUF, dequantizes to <paramref name="targetDtype"/>, and runs <paramref name="converter"/> on the resulting dict. Returns the converted result + the loader handle (caller disposes when finished using the tensors — many converters wrap mmap-backed tensors).</summary>
    public static (TConverted converted, GgufModelLoader.LoadedGgufModel handle) LoadGguf<TConverted>(
        string ggufPath,
        DType targetDtype,
        Func<Dictionary<string, Tensor>, TConverted> converter)
        where TConverted : class
    {
        if (converter is null) throw new ArgumentNullException(nameof(converter));

        (Dictionary<string, Tensor> weights, GgufModelLoader.LoadedGgufModel handle) =
            GgufModelLoader.LoadDequantized(ggufPath, targetDtype);
        try
        {
            TConverted converted = converter(weights);
            return (converted, handle);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }
}
