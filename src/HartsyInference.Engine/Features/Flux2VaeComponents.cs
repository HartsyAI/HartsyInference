using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
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
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            Tensor rawMean = Require(weights, BnMeanKey, vaePath);
            Tensor rawVar = Require(weights, BnVarKey, vaePath);
            VaeDecoder decoder = new VaeDecoder(VaeConfig.Flux2);
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
