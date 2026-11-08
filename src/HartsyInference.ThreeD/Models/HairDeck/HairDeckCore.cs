using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The HairDeck core: the image card DiT, the card VAE decoder, the optional colour head and chain DiT, and the
/// token conversions between them. Given the DINOv3 tokens of a crop it produces the decoded cards and chains of one sample.
/// The weights are borrowed from memory-mapped checkpoints that this instance keeps open until it is disposed.</summary>
public sealed unsafe class HairDeckCore : IDisposable
{
    /// <summary>Card slots of the token set.</summary>
    public const int CardSlots = 192;

    /// <summary>Channels of a card token.</summary>
    public const int CardDim = HairDeckTokenizer.TokenDim;

    /// <summary>Chain slots of the chain token set.</summary>
    public const int ChainSlots = 48;

    /// <summary>Channels of a chain token.</summary>
    public const int ChainDim = HairDeckChainDecoder.TokenDim;

    /// <summary>DINOv3 context tokens of the 448 px crop (CLS, four storage tokens and 28 x 28 patches).</summary>
    public const int DinoTokens = 789;

    /// <summary>DINOv3 ViT-L/16 token width.</summary>
    public const int DinoWidth = 1024;

    /// <summary>Euler steps and guidance of the chain DiT (fixed, as in the reference pipeline).</summary>
    public const int ChainSteps = 50;

    /// <summary>Classifier-free guidance scale of the chain DiT.</summary>
    public const float ChainGuidance = 2f;

    private const int ColorChannels = 3 * HairDeckColorHead.Bins;
    private const int VaeOutputChannels = HairDeckCardVaeCheckpointConverter.OutputChannels;
    private const string MetadataKey = "hairdeck";

    private readonly List<SafeTensorsLoader> _loaders;
    private readonly HairDeckImageDit _image;
    private readonly HairDeckCardVae _vae;
    private readonly HairDeckTokenizer _tokenizer;
    private readonly HairDeckColorHead? _color;
    private readonly HairDeckChainDit? _chain;
    private readonly float[] _chainMu;
    private readonly float[] _chainSd;
    private int _disposed;

    private HairDeckCore(List<SafeTensorsLoader> loaders, HairDeckImageDit image, HairDeckCardVae vae, HairDeckTokenizer tokenizer,
        HairDeckColorHead? color, HairDeckChainDit? chain, float[] chainMu, float[] chainSd)
    {
        _loaders = loaders;
        _image = image;
        _vae = vae;
        _tokenizer = tokenizer;
        _color = color;
        _chain = chain;
        _chainMu = chainMu;
        _chainSd = chainSd;
    }

    /// <summary>Whether the colour model is loaded (cards then carry colours when a crop and matte are given).</summary>
    public bool HasColor => _color is not null;

    /// <summary>Whether the chain DiT is loaded.</summary>
    public bool HasChains => _chain is not null;

    /// <summary>Loads the model set. Every checkpoint is validated against the others (token width, context length, latent width).</summary>
    public static HairDeckCore Load(HairDeckModelFiles files)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (files.ChainModel is not null && files.ChainStats is null)
            throw new ArgumentException("A chain model needs its chain statistics.", nameof(files));

        List<SafeTensorsLoader> loaders = [];
        bool ok = false;
        try
        {
            SafeTensorsLoader imageLoader = Open(files.ImageModel, loaders);
            HairDeckImageDit image = new(HairDeckImageCheckpointConverter.Convert(imageLoader.GetAllTensors()));
            if (image.Core.TokenDim != CardDim || image.ContextTokens != DinoTokens || image.DinoWidth != DinoWidth)
                throw new InvalidDataException($"HairDeck image model expects {image.Core.TokenDim}-channel tokens and {image.ContextTokens} context tokens of width {image.DinoWidth}.");

            SafeTensorsLoader vaeLoader = Open(files.CardVae, loaders);
            bool integrate = ReadIntegrate(vaeLoader);
            HairDeckCardVaeCheckpointConverter.Weights vaeWeights = HairDeckCardVaeCheckpointConverter.Convert(vaeLoader.GetAllTensors(), integrate);
            HairDeckStatsReader.Vectors output = HairDeckStatsReader.ReadOutput(files.CardVaeStats, VaeOutputChannels);
            HairDeckCardVae vae = new(vaeWeights, output.Mu, output.Sd);
            if (vae.LatentDim != HairDeckTokenizer.LatentDim)
                throw new InvalidDataException($"HairDeck card VAE latent width {vae.LatentDim} does not match the token latent width {HairDeckTokenizer.LatentDim}.");

            (float[] rootMu, float[] rootSd) = HairDeckStatsReader.ReadRoot(files.TokenStats);
            HairDeckStatsReader.Vectors latent = HairDeckStatsReader.ReadLatent(files.LatentStats, HairDeckTokenizer.LatentDim);
            HairDeckTokenizer tokenizer = new(rootMu, rootSd, latent.Mu, latent.Sd);

            HairDeckColorHead? color = null;
            if (files.ColorModel is not null)
            {
                SafeTensorsLoader colorLoader = Open(files.ColorModel, loaders);
                color = new HairDeckColorHead(HairDeckColorCheckpointConverter.Convert(colorLoader.GetAllTensors()));
                if (color.Core.Dim != image.Core.Dim || color.Core.ContextTokens != image.ContextTokens)
                    throw new InvalidDataException("HairDeck colour model does not share the image model's width and context.");
            }

            HairDeckChainDit? chain = null;
            float[] chainMu = [], chainSd = [];
            if (files.ChainModel is not null)
            {
                SafeTensorsLoader chainLoader = Open(files.ChainModel, loaders);
                chain = new HairDeckChainDit(HairDeckChainCheckpointConverter.Convert(chainLoader.GetAllTensors()));
                if (chain.Core.TokenDim != ChainDim || chain.CardSlots != CardSlots)
                    throw new InvalidDataException($"HairDeck chain model expects {chain.Core.TokenDim}-channel chains and {chain.CardSlots} card slots.");
                HairDeckStatsReader.Vectors stats = HairDeckStatsReader.ReadChain(files.ChainStats!, ChainDim - 1);
                chainMu = stats.Mu;
                chainSd = stats.Sd;
            }

            ok = true;
            return new HairDeckCore(loaders, image, vae, tokenizer, color, chain, chainMu, chainSd);
        }
        finally
        {
            if (!ok)
                foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
        }
    }

    /// <summary>Samples one hairstyle from the initial noise drawn for <paramref name="seed"/>. The DINOv3 tokens are
    /// <c>[789, 1024]</c>; the colour inputs are the crop <c>[3, 448, 448]</c> in 0..255 and its person matte <c>[448, 448]</c>
    /// (0 or 1), both or neither. The caller owns the tensors it passes.</summary>
    public HairDeckResult Generate(IBackend backend, Tensor dinoTokens, Tensor? colorImage, Tensor? colorMatte, int seed,
        int steps = 50, float cfg = 2f, bool withChains = true)
        => Run(backend, dinoTokens, colorImage, colorMatte, seed, steps, cfg, withChains, cardNoise: null, chainNoise: null);

    /// <summary>As <see cref="Generate"/> with explicit initial noise <c>[192, 40]</c> for the cards and <c>[48, 27]</c> for the
    /// chains (the reference's injected noise); the seed is only reported.</summary>
    public HairDeckResult GenerateWithNoise(IBackend backend, Tensor dinoTokens, Tensor? colorImage, Tensor? colorMatte, Tensor cardNoise,
        Tensor? chainNoise, int steps = 50, float cfg = 2f, bool withChains = true, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(cardNoise);
        return Run(backend, dinoTokens, colorImage, colorMatte, seed, steps, cfg, withChains, cardNoise, chainNoise);
    }

    /// <summary>Samples the card tokens <c>[192, 40]</c> from the given noise; the caller disposes the result.</summary>
    internal Tensor SampleCardTokens(IBackend backend, Tensor dinoTokens, Tensor cardNoise, int steps, float cfg)
    {
        ThrowIfDisposed();
        using Tensor context = _image.EncodeContext(backend, dinoTokens);
        return HairDeckEulerSampler.Sample(backend, _image.Core, context, cardNoise, steps, cfg);
    }

    private HairDeckResult Run(IBackend backend, Tensor dino, Tensor? colorImage, Tensor? colorMatte, int seed, int steps, float cfg,
        bool withChains, Tensor? cardNoise, Tensor? chainNoise)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(dino);
        if ((colorImage is null) != (colorMatte is null))
            throw new ArgumentException("The colour image and matte are given together or not at all.");

        Tensor? ownCardNoise = cardNoise is null ? HairDeckNoise.Create(seed, CardSlots, CardDim) : null;
        try
        {
            using Tensor x = SampleCardTokens(backend, dino, cardNoise ?? ownCardNoise!, steps, cfg);
            List<PendingCard> pending = DecodeCards(backend, x);
            if (pending.Count == 0)
                return new HairDeckResult([], [], seed);

            bool wantColor = _color is not null && colorImage is not null && colorMatte is not null;
            bool wantChains = withChains && _chain is not null;
            if (!wantColor && !wantChains)
                return new HairDeckResult(BuildCards(pending, null), [], seed);

            float[]?[]? colors = wantColor ? PredictColors(backend, dino, colorImage!, colorMatte!, x, pending) : null;
            IReadOnlyList<DecodedChain> chains = wantChains ? SampleChains(backend, x, seed, chainNoise) : [];
            return new HairDeckResult(BuildCards(pending, colors), chains, seed);
        }
        finally
        {
            ownCardNoise?.Dispose();
        }
    }

    private List<PendingCard> DecodeCards(IBackend backend, Tensor x)
    {
        float[] root = new float[3], latent = new float[HairDeckTokenizer.LatentDim], normal = new float[3];
        List<PendingCard> cards = [];
        for (int slot = 0; slot < CardSlots; slot++)
        {
            _tokenizer.DecodeSlot(x, slot, root, latent, normal, out int shapeClass, out bool exists);
            if (!exists) continue;
            float length = MathF.Sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            float[] direction = [normal[0] / (length + 1e-12f), normal[1] / (length + 1e-12f), normal[2] / (length + 1e-12f)];
            cards.Add(new PendingCard(slot, _vae.DecodeCard(backend, latent, root), direction, shapeClass));
        }
        return cards;
    }

    private float[]?[] PredictColors(IBackend backend, Tensor dino, Tensor colorImage, Tensor colorMatte, Tensor x,
        List<PendingCard> pending)
    {
        using Tensor clean = _tokenizer.Clean(x);
        using Tensor signed = _tokenizer.SignNormals(clean);
        using Tensor context = _color!.EncodeContext(backend, dino, colorImage, colorMatte);
        using Tensor lab = _color.PredictLab(backend, signed, context);
        float* labValues = (float*)lab.DataPointer;
        float[]?[] colors = new float[]?[pending.Count];
        float[] linear = new float[ColorChannels];
        for (int i = 0; i < pending.Count; i++)
        {
            ReadOnlySpan<float> row = new(labValues + (long)pending[i].Slot * ColorChannels, ColorChannels);
            HairDeckColorSpace.LabToLinearRgb(row, linear);
            colors[i] = (float[])linear.Clone();
        }
        return colors;
    }

    private IReadOnlyList<DecodedChain> SampleChains(IBackend backend, Tensor x, int seed, Tensor? chainNoise)
    {
        Tensor? ownNoise = chainNoise is null ? HairDeckNoise.Create(seed, ChainSlots, ChainDim) : null;
        try
        {
            using Tensor clean = _tokenizer.Clean(x);
            using Tensor context = _chain!.EncodeContext(backend, clean);
            using Tensor xc = HairDeckEulerSampler.Sample(backend, _chain.Core, context, chainNoise ?? ownNoise!, ChainSteps, ChainGuidance);
            return HairDeckChainDecoder.Decode(xc, _chainMu, _chainSd);
        }
        finally
        {
            ownNoise?.Dispose();
        }
    }

    private static List<DecodedCard> BuildCards(List<PendingCard> pending, float[]?[]? colors)
    {
        List<DecodedCard> cards = new(pending.Count);
        for (int i = 0; i < pending.Count; i++)
        {
            PendingCard p = pending[i];
            int n = p.Points.Length / VaeOutputChannels;
            float[] centerline = new float[n * 3], width = new float[n], thickness = new float[n];
            double sumWidth = 0, sumThickness = 0;
            for (int j = 0; j < n; j++)
            {
                for (int c = 0; c < 3; c++) centerline[j * 3 + c] = p.Points[j * VaeOutputChannels + c];
                width[j] = p.Points[j * VaeOutputChannels + 3];
                thickness[j] = p.Points[j * VaeOutputChannels + 4];
                sumWidth += width[j];
                sumThickness += thickness[j];
            }
            cards.Add(new DecodedCard(centerline, n, (float)(sumWidth / n), (float)(sumThickness / n), p.ShapeClass, colors?[i])
            {
                WidthProfile = width,
                ThicknessProfile = thickness,
                RootNormal = p.RootNormal,
            });
        }
        return cards;
    }

    private static SafeTensorsLoader Open(string path, List<SafeTensorsLoader> loaders)
    {
        SafeTensorsLoader loader = new();
        loaders.Add(loader);
        loader.Load(path);
        return loader;
    }

    private static bool ReadIntegrate(SafeTensorsLoader loader)
    {
        if (loader.Metadata is null || !loader.Metadata.TryGetValue(MetadataKey, out string? json))
            return false;
        using JsonDocument doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("args", out JsonElement args))
            return false;
        if (args.TryGetProperty("dec", out JsonElement dec) && dec.GetString() != "ada")
            throw new NotSupportedException($"HairDeck card VAE decoder '{dec.GetString()}' is not supported; the released decoder is 'ada'.");
        return args.TryGetProperty("integ", out JsonElement integ) && integ.ValueKind == JsonValueKind.True;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases the checkpoint mappings; the tensors of this instance must not be used afterwards.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (SafeTensorsLoader loader in _loaders) loader.Dispose();
        _loaders.Clear();
    }

    private sealed record PendingCard(int Slot, float[] Points, float[] RootNormal, int ShapeClass);
}
