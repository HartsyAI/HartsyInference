using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>A loaded HairDeck image model: the DINOv3 ViT-L/16 encoder at 448 px that conditions it, and the
/// <see cref="HairDeckCore"/> that samples the cards, colours and chains.</summary>
public sealed class HairDeckModel : IDisposable
{
    private const int Resolution = HairDeckInput.Resolution;
    private const int Plane = Resolution * Resolution;
    private const int DinoTokens = HairDeckCore.DinoTokens;
    private const int DinoWidth = HairDeckCore.DinoWidth;

    private static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];

    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _dinoLoader;
    private readonly Dinov3VisionEncoder _dino;
    private readonly HairDeckCore _core;
    private int _disposed;

    private HairDeckModel(IBackend backend, SafeTensorsLoader dinoLoader, Dinov3VisionEncoder dino, HairDeckCore core)
    {
        _backend = backend;
        _dinoLoader = dinoLoader;
        _dino = dino;
        _core = core;
    }

    /// <summary>Whether the colour model is loaded.</summary>
    public bool HasColor => _core.HasColor;

    /// <summary>Whether the chain model is loaded.</summary>
    public bool HasChains => _core.HasChains;

    /// <summary>Loads the HairDeck model set from <paramref name="modelRoot"/> (the folder holding <c>config.json</c>, <c>ckpt/</c> and
    /// <c>data/</c>) and the DINOv3 encoder from <paramref name="dinoWeightsPath"/>.</summary>
    /// <exception cref="FileNotFoundException">A required checkpoint or statistics file is missing.</exception>
    public static HairDeckModel Load(IBackend backend, string modelRoot, string dinoWeightsPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentException.ThrowIfNullOrEmpty(modelRoot);
        ArgumentException.ThrowIfNullOrEmpty(dinoWeightsPath);

        HairDeckModelFiles files = ResolveFiles(modelRoot);
        SafeTensorsLoader dinoLoader = new();
        try
        {
            dinoLoader.Load(dinoWeightsPath);
            Dictionary<string, Tensor> dinoWeights = dinoLoader.GetAllTensors();
            Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(dinoWeights, Resolution);
            Dinov3VisionEncoder dino = new(preset);
            dino.LoadWeights(dinoWeights);
            HairDeckCore core = HairDeckCore.Load(files);
            return new HairDeckModel(backend, dinoLoader, dino, core);
        }
        catch
        {
            dinoLoader.Dispose();
            throw;
        }
    }

    /// <summary>Generates one hairstyle for <paramref name="input"/>: the DINOv3 tokens of the crop, then the image model, the
    /// colour model when <see cref="HairDeckOptions.Colors"/> is set and the chain model when <see cref="HairDeckOptions.Chains"/> is set.</summary>
    /// <exception cref="InvalidOperationException">A requested component is not loaded; no partial result is returned.</exception>
    public HairDeckResult Generate(HairDeckInput input, HairDeckOptions options)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(input);
        options = options.Validate();
        if (options.Seed < 0 || options.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(options), options.Seed, "The HairDeck seed must be between 0 and int.MaxValue.");
        if (options.Colors && !_core.HasColor)
            throw new InvalidOperationException("Colours were requested but the HairDeck colour model is not loaded; no partial result is returned.");
        if (options.Chains && !_core.HasChains)
            throw new InvalidOperationException("Chains were requested but the HairDeck chain model is not loaded; no partial result is returned.");

        using Tensor dino = EncodeCrop(input.Rgb448);
        using Tensor? colorImage = options.Colors ? ColorImage(input.Rgb448) : null;
        using Tensor? colorMatte = options.Colors ? ColorMatte(input.Matte448) : null;
        return _core.Generate(_backend, dino, colorImage, colorMatte, (int)options.Seed,
            options.Steps, (float)options.Cfg, options.Chains);
    }

    /// <summary>DINOv3 tokens <c>[789, 1024]</c> of the crop. The reference feeds uint8 and the encoder applies the ImageNet
    /// normalisation, so the crop is scaled to [0, 1] and normalised here.</summary>
    private Tensor EncodeCrop(byte[] rgb)
    {
        using Tensor pixels = new(new TensorShape(1, 3, Resolution, Resolution), DType.F32);
        Span<float> values = pixels.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < Plane; i++)
            {
                values[c * Plane + i] = (rgb[c * Plane + i] / 255f - ImageNetMean[c]) / ImageNetStd[c];
            }
        }

        using Tensor encoded = _dino.Encode(_backend, pixels);
        Tensor tokens = new(new TensorShape(DinoTokens, DinoWidth), DType.F32);
        encoded.AsSpan<float>()[..(DinoTokens * DinoWidth)].CopyTo(tokens.AsSpan<float>());
        return tokens;
    }

    /// <summary>The crop as <c>[3, 448, 448]</c> in 0..255, the colour model's image input.</summary>
    private static Tensor ColorImage(byte[] rgb)
    {
        Tensor image = new(new TensorShape(3, Resolution, Resolution), DType.F32);
        Span<float> values = image.AsSpan<float>();
        for (int i = 0; i < 3 * Plane; i++)
        {
            values[i] = rgb[i];
        }

        return image;
    }

    /// <summary>The person matte as <c>[448, 448]</c> with 0 or 1, the colour model's mask input.</summary>
    private static Tensor ColorMatte(byte[] matte)
    {
        Tensor mask = new(new TensorShape(Resolution, Resolution), DType.F32);
        Span<float> values = mask.AsSpan<float>();
        for (int i = 0; i < Plane; i++)
        {
            values[i] = matte[i] >= 128 ? 1f : 0f;
        }

        return mask;
    }

    /// <summary>Maps the model root onto the checkpoint layout. Required files must exist; the colour and chain models are
    /// optional and are null when their files are absent (the request then fails rather than dropping them).</summary>
    private static HairDeckModelFiles ResolveFiles(string root)
    {
        string Required(string relative)
        {
            string path = Path.Combine(root, relative);
            if (!File.Exists(path))
                throw new FileNotFoundException($"HairDeck needs '{relative}' under '{root}'.", path);
            return path;
        }

        string? Optional(string relative)
        {
            string path = Path.Combine(root, relative);
            return File.Exists(path) ? path : null;
        }

        return new HairDeckModelFiles(
            ImageModel: Required(Path.Combine("ckpt", "image_model", "model.safetensors")),
            CardVae: Required(Path.Combine("ckpt", "card_vae", "model.safetensors")),
            CardVaeStats: Required(Path.Combine("ckpt", "card_vae", "stats.json")),
            TokenStats: Required(Path.Combine("data", "token_stats_head.json")),
            LatentStats: Required(Path.Combine("ckpt", "card_vae", "latent_stats_head.json")),
            ColorModel: Optional(Path.Combine("ckpt", "color_model", "model.safetensors")),
            ChainModel: Optional(Path.Combine("ckpt", "chain_model", "model.safetensors")),
            ChainStats: Optional(Path.Combine("data", "spring", "chain_stats_head.json")));
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _core.Dispose();
        _dinoLoader.Dispose();
    }
}
