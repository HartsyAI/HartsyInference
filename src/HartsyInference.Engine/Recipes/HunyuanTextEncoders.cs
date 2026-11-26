using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Recipes;

/// <summary>The two text encoders the HunyuanImage and HunyuanVideo-1.5 families share: Qwen2.5-VL-7B (the fp8-scaled side file, read through the
/// HunyuanImage conditioning path) and Glyph-ByT5-small for quoted text. Both recipes load them from the same side files the same way.</summary>
public static class HunyuanTextEncoders
{
    /// <summary>Loads the Qwen2.5-VL-7B encoder. Its container is added to <paramref name="loaders"/>, which owns it from then on.</summary>
    public static (LlamaStyleEncoder Llama, HunyuanImageQwenTextEncoder Encoder) LoadQwen25Vl(RecipeContext context, List<IDisposable> loaders)
    {
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen25Vl7BHunyuan, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenPath);
        loaders.Add(qwenLoader);
        LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
        llama.LoadWeights(qwenLoader.GetAllTensors());
        return (llama, new HunyuanImageQwenTextEncoder(llama));
    }

    /// <summary>Loads Glyph-ByT5-small for the quoted text of a prompt. A recipe calls it on the first prompt that quotes text. The caller owns the
    /// returned encoder and its container; a failed load releases the container here.</summary>
    public static (T5TextEncoder Encoder, IDisposable Loader) LoadGlyphByt5(CancellationToken cancel)
    {
        string byt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.ByT5SmallGlyphXl, onProgress: null, cancel).GetAwaiter().GetResult();
        SafeTensorsLoader byt5Loader = new SafeTensorsLoader();
        try
        {
            byt5Loader.Load(byt5Path);
            T5TextEncoder byt5 = new T5TextEncoder(T5TextEncoderConfig.ByT5SmallGlyph);
            byt5.LoadWeights(byt5Loader.GetAllTensors());
            Logs.Info("[HunyuanTextEncoders] Glyph-ByT5-small loaded for quoted prompt text.");
            return (byt5, byt5Loader);
        }
        catch
        {
            // A failed load must not leave the container mapped: nothing else holds it yet.
            byt5Loader.Dispose();
            throw;
        }
    }
}
