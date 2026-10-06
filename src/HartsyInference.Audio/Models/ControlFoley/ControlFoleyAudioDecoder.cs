using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>ControlFoley's audio decoder: generator latent <c>[T, latent_dim]</c> to mono 44.1 kHz PCM through the
/// VAE decoder (<c>FeaturesUtils.decode</c>) and the BigVGAN-v2 vocoder (<c>FeaturesUtils.vocode</c>).</summary>
public sealed class ControlFoleyAudioDecoder : IDisposable
{
    private readonly ControlFoleyVaeConfig _vaeConfig;
    private readonly ControlFoleyVaeDecoder _vae;
    private readonly ControlFoleyBigVgan _vocoder;
    private readonly List<IDisposable> _sources = [];
    private int _disposed;

    public ControlFoleyAudioDecoder(ControlFoleyVaeConfig? vaeConfig = null, ControlFoleyBigVganConfig? vocoderConfig = null)
    {
        _vaeConfig = vaeConfig ?? ControlFoleyVaeConfig.V44k;
        _vae = new ControlFoleyVaeDecoder(_vaeConfig);
        _vocoder = new ControlFoleyBigVgan(vocoderConfig ?? ControlFoleyBigVganConfig.V44k);
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRate => 44_100;

    /// <summary>Latent channels per frame the decoder expects.</summary>
    public int LatentDim => _vaeConfig.EmbedDim;

    /// <summary>Loads from already-materialised tensors; the caller keeps them alive while the decoder is in use.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> vae, IReadOnlyDictionary<string, Tensor> vocoder,
        string vaePrefix = "", string vocoderPrefix = "")
    {
        _vae.LoadWeights(vae, vaePrefix);
        _vocoder.LoadWeights(vocoder, vocoderPrefix);
    }

    /// <summary>Loads the released <c>v1-44.pth</c> VAE and the <c>bigvgan_generator.pt</c> vocoder, or the same tensors
    /// converted to <c>.safetensors</c>.</summary>
    public void LoadFromFiles(string vaePath, string vocoderPath)
    {
        Dictionary<string, Tensor> vae = Open(vaePath);
        Dictionary<string, Tensor> vocoder = Open(vocoderPath);
        LoadWeights(vae, vocoder);
    }

    /// <summary>Loads one <c>.safetensors</c> holding both models under the <c>vae.</c> and <c>voc.</c> prefixes.</summary>
    public void LoadFromCombinedFile(string path)
    {
        Dictionary<string, Tensor> all = Open(path);
        LoadWeights(all, all, "vae.", "voc.");
    }

    /// <summary>Decodes a flat <c>[T, LatentDim]</c> latent to <c>T * 2 * 512</c> samples in [-1, 1].</summary>
    public float[] Decode(IBackend backend, float[] latent)
    {
        using Tensor mel = DecodeToMel(backend, latent);
        return Vocode(backend, mel);
    }

    /// <summary>Decodes a flat <c>[T, LatentDim]</c> latent to the mel <c>[1, DataDim, 2T]</c>; the caller owns the result.</summary>
    public Tensor DecodeToMel(IBackend backend, float[] latent, Action<string, Tensor>? tap = null)
    {
        ArgumentNullException.ThrowIfNull(latent);
        int dim = _vaeConfig.EmbedDim;
        if (latent.Length == 0 || latent.Length % dim != 0)
            throw new ArgumentException($"Latent length {latent.Length} is not a positive multiple of {dim}.", nameof(latent));
        int t = latent.Length / dim;
        using Tensor frames = new(new TensorShape(t, dim), DType.F32);
        latent.AsSpan().CopyTo(frames.AsSpan<float>());
        using Tensor z = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(z, frames, t, dim);
        return _vae.Decode(backend, z, tap);
    }

    /// <summary>Vocodes a flat mel <c>[NumMels, frames]</c> (as <see cref="ControlFoleyMelConverter.Compute"/> returns it).</summary>
    public float[] Vocode(IBackend backend, float[] mel, int frames)
    {
        ArgumentNullException.ThrowIfNull(mel);
        if (frames < 1 || mel.Length % frames != 0) throw new ArgumentException("mel length is not a multiple of frames.", nameof(mel));
        using Tensor t = new(new TensorShape(1, mel.Length / frames, frames), DType.F32);
        mel.AsSpan().CopyTo(t.AsSpan<float>());
        return Vocode(backend, t);
    }

    /// <summary>Vocodes a mel tensor <c>[1, NumMels, T]</c>.</summary>
    public float[] Vocode(IBackend backend, Tensor mel, Action<string, Tensor>? tap = null)
    {
        using Tensor wave = _vocoder.Forward(backend, mel, tap);
        return wave.AsSpan<float>().ToArray();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _vae.Dispose();
        _vocoder.Dispose();
        foreach (IDisposable source in _sources) source.Dispose();
        GC.SuppressFinalize(this);
    }

    private Dictionary<string, Tensor> Open(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader loader = new();
            _sources.Add(loader);
            loader.Load(path);
            return loader.GetAllTensors();
        }
        PytorchPickleLoader pickle = new();
        _sources.Add(pickle);
        pickle.Load(path);
        return pickle.GetAllTensors();
    }
}
