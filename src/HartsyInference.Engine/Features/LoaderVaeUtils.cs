using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Shared staging for the FLUX.1 16-channel autoencoder (<c>ae.safetensors</c>, original LDM naming): remaps every key to the diffusers naming <c>VaeDecoder</c> expects and upcasts 16-bit weights to F32, plus the encoder-half construction every img2img-capable recipe needs. <para>The encoder and decoder halves live in one weight dictionary, so a recipe that already staged weights for its decoder builds its encoder from the same object at no extra I/O — see <c>SdxlRecipe</c> for the canonical shape.</para></summary>
public static class LoaderVaeUtils
{
    /// <summary>The key every diffusers-named VAE encoder starts from; its absence means the file is decode-only.</summary>
    private const string VaeEncoderSentinelKey = "encoder.conv_in.weight";

    /// <summary>The equivalent sentinel for the Qwen-Image 3D-causal VAE, whose encoder is <c>conv1</c>-rooted.</summary>
    private const string QwenEncoderSentinelKey = "encoder.conv1.weight";

    /// <summary>Whether <paramref name="weights"/> carries an encoder half at all. Some published checkpoints ship a decode-only VAE, which is a property of the file rather than of the family — recipes use this to decide between a hard failure and constructing an encoder-less pipeline that refuses img2img at generate time.</summary>
    public static bool HasEncoderWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return weights.ContainsKey(VaeEncoderSentinelKey);
    }

    /// <inheritdoc cref="HasEncoderWeights"/>
    public static bool HasQwenEncoderWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return weights.ContainsKey(QwenEncoderSentinelKey);
    }

    /// <summary>Builds the encoder matching a decoder already loaded from <paramref name="weights"/>, or throws naming <paramref name="family"/> when the file is decode-only — so a missing encoder half surfaces at load time rather than as a failed generation much later.</summary>
    public static VaeEncoder BuildEncoder(VaeConfig config, IReadOnlyDictionary<string, Tensor> weights, string family)
    {
        return TryBuildEncoder(config, weights, family)
            ?? throw new InvalidOperationException(
                $"[{family}] This checkpoint's VAE carries no encoder half (no '{VaeEncoderSentinelKey}'), so it cannot "
                + "encode an init image. Use a full VAE checkpoint, or supply one via the VAE component override.");
    }

    /// <summary>Builds the encoder matching a decoder already loaded from <paramref name="weights"/>, or returns null when the file is decode-only. Callers must pass the null through to the pipeline so img2img is refused by name; dropping it silently is the one failure mode that still looks like a successful generation.</summary>
    public static VaeEncoder? TryBuildEncoder(VaeConfig config, IReadOnlyDictionary<string, Tensor> weights, string family)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!HasEncoderWeights(weights))
        {
            Logs.Info($"[{family}] VAE is decode-only — img2img and inpaint will be refused for this checkpoint.");
            return null;
        }
        VaeEncoder encoder = new VaeEncoder(config);
        try
        {
            encoder.LoadWeights(weights);
            return encoder;
        }
        catch (Exception ex)
        {
            Logs.Error($"[{family}] VAE encoder weights failed to load.", ex);
            throw;
        }
    }

    /// <summary>The Qwen-Image 3D-causal counterpart of <see cref="TryBuildEncoder"/>. Kept separate because that VAE normalizes latents per channel (<c>latents_mean</c> / <c>latents_std</c>), which the generic <see cref="VaeEncoder"/> has no path for — families on <c>VaeConfig.QwenImage</c> must come through here.</summary>
    public static QwenImageVaeEncoder? TryBuildQwenEncoder(VaeConfig config, IReadOnlyDictionary<string, Tensor> weights, string family)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!HasQwenEncoderWeights(weights))
        {
            Logs.Info($"[{family}] Qwen-Image VAE is decode-only — img2img and inpaint will be refused for this checkpoint.");
            return null;
        }
        QwenImageVaeEncoder encoder = new QwenImageVaeEncoder(config);
        try
        {
            encoder.LoadWeights(weights);
            return encoder;
        }
        catch (Exception ex)
        {
            Logs.Error($"[{family}] Qwen-Image VAE encoder weights failed to load.", ex);
            throw;
        }
    }

    /// <summary>Loads a FLUX.1-family VAE file and returns diffusers-keyed F32 weights; keys already in diffusers naming pass through (ConvertVaeKey is identity-tolerant) and unknown keys are dropped. F32 tensors borrow their mapped storage from the returned loader; upcast tensors are caller-owned. Keep the loader alive while any borrowed tensor is used and dispose the distinct upcast tensors when their model owner is released.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadFluxVaeF32(string filePath)
    {
        ArgumentException.ThrowIfNullOrEmpty(filePath);
        SafeTensorsLoader loader = new SafeTensorsLoader();
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>();
        List<Tensor> ownedCasts = [];
        try
        {
            loader.Load(filePath);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(kvp.Key);
                if (diffusersKey is null)
                {
                    continue;
                }
                DType dt = kvp.Value.DType;
                Tensor staged = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
                if (!ReferenceEquals(staged, kvp.Value))
                    ownedCasts.Add(staged);
                if (result.TryGetValue(diffusersKey, out Tensor? replaced) && ownedCasts.Remove(replaced))
                    replaced.Dispose();
                result[diffusersKey] = staged;
            }
            return (result, loader);
        }
        catch (Exception ex)
        {
            HartsyInference.Core.Logging.Logs.Error($"[Features][VAE] Failed to stage Flux VAE weights from '{filePath}'.", ex);
            foreach (Tensor tensor in ownedCasts)
                tensor.Dispose();
            loader.Dispose();
            throw;
        }
    }
}
