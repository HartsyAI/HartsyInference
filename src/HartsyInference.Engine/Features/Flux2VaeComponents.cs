using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>The Flux.2 VAE as every family on it consumes it (Flux.2, FLUX.3): the decoder, the encoder when the file
/// ships one, and the patchified-latent BatchNorm <c>running_mean</c>/<c>running_var</c> as F32 <c>[128]</c>, which the
/// pipelines read through raw pointers. Owns the file mapping and any widened stat copies.</summary>
public sealed class Flux2VaeComponents : IDisposable
{
    private const string BnMeanKey = "bn.running_mean";
    private const string BnVarKey = "bn.running_var";

    private readonly SafeTensorsLoader _loader;
    private readonly bool _ownsMean;
    private readonly bool _ownsVar;
    private int _disposed;

    private Flux2VaeComponents(SafeTensorsLoader loader, VaeDecoder decoder, VaeEncoder? encoder, Tensor bnMean, bool ownsMean,
        Tensor bnVar, bool ownsVar)
    {
        _loader = loader;
        Decoder = decoder;
        Encoder = encoder;
        BnMean = bnMean;
        _ownsMean = ownsMean;
        BnVar = bnVar;
        _ownsVar = ownsVar;
    }

    /// <summary>The 32-channel decoder.</summary>
    public VaeDecoder Decoder { get; }

    /// <summary>The encoder, or null for a decode-only file.</summary>
    public VaeEncoder? Encoder { get; }

    /// <summary>F32 <c>[128]</c> BatchNorm running mean of the patchified latent.</summary>
    public Tensor BnMean { get; }

    /// <summary>F32 <c>[128]</c> BatchNorm running variance of the patchified latent.</summary>
    public Tensor BnVar { get; }

    /// <summary>Loads <paramref name="vaePath"/>, refusing a file without the BatchNorm stats (a Flux.1 <c>ae</c>).</summary>
    public static Flux2VaeComponents Load(string vaePath, string family)
    {
        ArgumentException.ThrowIfNullOrEmpty(vaePath);
        SafeTensorsLoader loader = new SafeTensorsLoader();
        try
        {
            loader.Load(vaePath);
            Dictionary<string, Tensor> weights = ToDiffusersLayout(loader.GetAllTensors());
            Tensor rawMean = Require(weights, BnMeanKey, vaePath);
            Tensor rawVar = Require(weights, BnVarKey, vaePath);
            VaeDecoder decoder = new VaeDecoder(WithCheckpointDecoder(VaeConfig.Flux2, weights));
            decoder.LoadWeights(weights);
            VaeEncoder? encoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux2, weights, family);
            Tensor mean = TensorCasts.EnsureF32(rawMean);
            Tensor var = TensorCasts.EnsureF32(rawVar);
            return new Flux2VaeComponents(loader, decoder, encoder, mean, !ReferenceEquals(mean, rawMean),
                var, !ReferenceEquals(var, rawVar));
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>The Comfy-Org repack of the FLUX.2 VAE (Fibo 2.0's <c>vae/flux2-vae.safetensors</c>) keeps ComfyUI's native ldm
    /// names: <c>decoder.mid.attn_1.q</c>, <c>decoder.up.N.block</c>, <c>decoder.post_quant_conv</c>, <c>encoder.quant_conv</c>, and
    /// 1x1-conv attention projections <c>[C, C, 1, 1]</c>. The decoder and encoder read the diffusers layout, so such a file is
    /// renamed (attention projections reshaped to <c>[C, C]</c> views) and a diffusers-layout file passes through unchanged.</summary>
    internal static Dictionary<string, Tensor> ToDiffusersLayout(Dictionary<string, Tensor> weights)
    {
        if (!weights.ContainsKey("decoder.mid.attn_1.q.weight"))
            return weights;
        Dictionary<string, Tensor> converted = new(weights.Count);
        foreach ((string key, Tensor tensor) in weights)
        {
            string? mapped = key switch
            {
                _ when key.StartsWith("bn.", StringComparison.Ordinal) => key,
                _ when key.StartsWith("decoder.post_quant_conv.", StringComparison.Ordinal) => key["decoder.".Length..],
                _ when key.StartsWith("encoder.quant_conv.", StringComparison.Ordinal) => key["encoder.".Length..],
                _ => CheckpointConvertUtils.ConvertVaeKey(key),
            };
            if (mapped is null)
                continue;
            bool attentionProjection = mapped.EndsWith(".weight", StringComparison.Ordinal) && tensor.Shape.Rank == 4
                && (mapped.Contains(".attentions.", StringComparison.Ordinal));
            converted[mapped] = attentionProjection
                ? tensor.Reshape(new TensorShape(tensor.Shape[0], tensor.Shape[1]))
                : tensor;
        }
        return converted;
    }

    /// <summary>The decoder's stage widths as the checkpoint builds them: the up-blocks' output channels, reversed. FLUX.2's
    /// own decoder mirrors the encoder (128-512); OpenMOSS's MOSS-Image folder ships a narrower one (96-384) over the same
    /// latent, so a fixed preset would load its weights into the wrong layers.</summary>
    internal static VaeConfig WithCheckpointDecoder(VaeConfig config, Dictionary<string, Tensor> weights)
    {
        int stages = config.BlockOutChannels.Length;
        int[] decoder = new int[stages];
        for (int up = 0; up < stages; up++)
        {
            if (!weights.TryGetValue($"decoder.up_blocks.{up}.resnets.0.conv2.weight", out Tensor? conv))
                return config;
            decoder[stages - 1 - up] = (int)conv.Shape[0];
        }
        for (int i = 0; i < stages; i++)
        {
            if (decoder[i] != config.BlockOutChannels[i])
                return config with { DecoderBlockOutChannels = decoder };
        }
        return config;
    }

    private static Tensor Require(Dictionary<string, Tensor> weights, string key, string path) =>
        weights.TryGetValue(key, out Tensor? tensor)
            ? tensor
            : throw new InvalidOperationException(
                $"Flux.2 VAE '{Path.GetFileName(path)}' is missing '{key}'. Verify this is a Flux.2 VAE, not a Flux.1 ae.safetensors.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsMean) BnMean.Dispose();
        if (_ownsVar) BnVar.Dispose();
        _loader.Dispose();
    }
}
