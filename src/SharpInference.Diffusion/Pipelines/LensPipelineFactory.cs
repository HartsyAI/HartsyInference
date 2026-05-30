using SharpInference.Core.Backends;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Wires <see cref="LensCheckpointConverter"/> output into a ready-to-run <see cref="LensPipeline"/>.
/// Mirrors the manual wiring that <c>Flux2GenerationTests</c> does for Flux.2: build the
/// <see cref="LensTransformer"/> + <see cref="LensGptOssEncoder"/> + Flux.2 <see cref="VaeDecoder"/>, call
/// each component's <c>LoadWeights</c>, and extract the patchified-latent BatchNorm <c>running_mean</c> /
/// <c>running_var</c> the Lens (and Flux.2) latent space is normalized in. Returns a
/// <see cref="LensPipelineBundle"/> so the caller has a single disposable handle over the pipeline, its
/// components, the BN copies, and any loaders opened here.
///
/// <para>The BN stats live at the top level of the VAE dict (<c>bn.running_mean</c> / <c>bn.running_var</c>)
/// alongside the module weights — same as Flux.2 Klein. They are cast to owned F32 <c>[128]</c> copies
/// because <see cref="LensPipeline"/> reads them through raw <c>float*</c> on the un-normalize path; the
/// originals (memory-mapped views, or BF16/F16 in the ComfyUI VAE file) are left untouched.</para></summary>
public static class LensPipelineFactory
{
    private const string BnMeanKey = "bn.running_mean";
    private const string BnVarKey = "bn.running_var";

    /// <summary>Builds a pipeline from already-converted per-component weights (e.g. from
    /// <see cref="LensCheckpointConverter.Convert"/> or <see cref="LensCheckpointConverter.ConvertComfy"/>).
    /// The caller retains ownership of the weight tensors inside <paramref name="weights"/>; the returned
    /// bundle owns only what this factory allocates (the BN F32 copies + the component shells).</summary>
    /// <param name="withTextEncoder">When <c>false</c>, skips loading the GPT-OSS encoder — the resulting
    /// pipeline supports <see cref="LensPipeline.GenerateFromEmbeddings"/> only.</param>
    public static LensPipelineBundle BuildFromConverted(IBackend backend,
        LensCheckpointConverter.ConvertedWeights weights, LensConfig config,
        bool withTextEncoder = true, float bnEps = 1e-5f)
        => Wire(backend, weights.Transformer, weights.TextEncoder, weights.Vae, config,
            withTextEncoder, bnEps, ownedLoaders: []);

    /// <summary>Loads + converts the three ComfyUI Lens files (<c>Comfy-Org/Lens</c>: DiT, GPT-OSS text
    /// encoder, Flux.2 VAE) and wires them into a pipeline. The bundle keeps the loaders alive — the
    /// passthrough VAE tensors are memory-mapped views into <paramref name="vaePath"/>.</summary>
    public static LensPipelineBundle LoadFromComfyFiles(IBackend backend,
        string ditPath, string textEncoderPath, string vaePath, LensConfig config,
        bool withTextEncoder = true, float bnEps = 1e-5f)
    {
        (LensCheckpointConverter.ConvertedWeights weights, SafeTensorsLoader[] loaders) =
            LensCheckpointConverter.LoadAndConvertComfy(ditPath, textEncoderPath, vaePath);
        try
        {
            return Wire(backend, weights.Transformer, weights.TextEncoder, weights.Vae, config,
                withTextEncoder, bnEps, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to wire Lens pipeline from ComfyUI files (dit={ditPath})", ex);
            for (int i = 0; i < loaders.Length; i++) loaders[i].Dispose();
            throw;
        }
    }

    /// <summary>Loads + converts a single-file Lens checkpoint (HF diffusers-native combined dump) and
    /// wires it into a pipeline. The bundle keeps the loader alive (passthrough tensors are mmap views).</summary>
    public static LensPipelineBundle LoadFromSingleFile(IBackend backend,
        string checkpointPath, LensConfig config, bool withTextEncoder = true, float bnEps = 1e-5f)
    {
        (LensCheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LensCheckpointConverter.LoadAndConvert(checkpointPath);
        try
        {
            return Wire(backend, weights.Transformer, weights.TextEncoder, weights.Vae, config,
                withTextEncoder, bnEps, [loader]);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to wire Lens pipeline from {checkpointPath}", ex);
            loader.Dispose();
            throw;
        }
    }

    private static LensPipelineBundle Wire(IBackend backend,
        Dictionary<string, Tensor> transformerWeights,
        Dictionary<string, Tensor> textEncoderWeights,
        Dictionary<string, Tensor> vaeWeights,
        LensConfig config, bool withTextEncoder, float bnEps, IDisposable[] ownedLoaders)
    {
        // Validate + extract the BN stats up front so a VAE missing them fails fast, before the
        // multi-GB transformer + encoder loads.
        (Tensor bnMean, Tensor bnVar) = ExtractBnStats(vaeWeights);

        LensTransformer transformer = new LensTransformer(config);
        transformer.LoadWeights(transformerWeights);

        LensGptOssEncoder? encoder = null;
        if (withTextEncoder)
        {
            encoder = new LensGptOssEncoder();
            encoder.LoadWeights(textEncoderWeights);
        }

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux2);
        vaeDecoder.LoadWeights(vaeWeights);

        LensPipeline pipeline = new LensPipeline(
            backend, transformer, encoder, vaeDecoder, bnMean, bnVar, config, bnEps);

        Logs.Info($"Lens pipeline wired: transformer={transformerWeights.Count} keys, " +
                  $"encoder={(withTextEncoder ? textEncoderWeights.Count : 0)} keys, " +
                  $"vae={vaeWeights.Count} keys, bn_mean={bnMean.Shape}, bn_var={bnVar.Shape}.");

        return new LensPipelineBundle(pipeline, transformer, encoder, vaeDecoder, bnMean, bnVar, ownedLoaders);
    }

    /// <summary>Pulls the BatchNorm running stats out of the VAE dict and returns owned F32 copies.</summary>
    private static (Tensor mean, Tensor var) ExtractBnStats(Dictionary<string, Tensor> vaeWeights)
    {
        if (!vaeWeights.TryGetValue(BnMeanKey, out Tensor? rawMean))
            throw new SharpInferenceException(
                $"Lens VAE weights missing '{BnMeanKey}' — the Flux.2 semantic VAE stores the patchified-latent " +
                "BatchNorm running stats at the top level; the latent space cannot be un-normalized without it.");
        if (!vaeWeights.TryGetValue(BnVarKey, out Tensor? rawVar))
            throw new SharpInferenceException(
                $"Lens VAE weights missing '{BnVarKey}' — see the '{BnMeanKey}' error above.");

        // CastTo(F32) always allocates an owned copy (F32→F32 included), so the bundle can dispose these
        // without touching the loader-owned / caller-owned source tensors.
        return (rawMean.CastTo(DType.F32), rawVar.CastTo(DType.F32));
    }
}
