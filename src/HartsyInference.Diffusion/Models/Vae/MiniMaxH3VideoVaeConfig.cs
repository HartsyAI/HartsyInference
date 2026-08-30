using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>MiniMax-H3 video VAE (<c>AutoencoderKLLegacy</c> with <c>use_vit_decoder</c>): a 3D causal CNN encoder
/// paired with a <b>ViT3D transformer decoder</b>, 16x spatial / 4x temporal, 24 latent channels, scale factor 1.0.
/// Only the decoder half is modelled by <see cref="MiniMaxH3VideoVaeDecoder"/>. Heads, head dim, rope theta/ratio and
/// eps are invisible in the checkpoint (the qk norms are non-affine and <c>to_qkv</c> is a flat <c>[3·dim, dim]</c>),
/// so they come from <see cref="FromJson"/> or the defaults; <see cref="Detect"/> recovers the rest from shapes.</summary>
public sealed record MiniMaxH3VideoVaeConfig
{
    /// <summary>Attention heads in the ViT decoder; not derivable from the checkpoint.</summary>
    public int Heads { get; init; } = 32;

    /// <summary>Per-head width; not derivable from the checkpoint.</summary>
    public int DimHead { get; init; } = 64;

    public int NumLayers { get; init; } = 36;

    /// <summary>Learned suffix tokens appended after the patch rows, followed by one zero "cls" row.</summary>
    public int NumRegisterTokens { get; init; } = 4;

    /// <summary>Gated-SiLU FFN width multiplier: <c>w1</c> emits <c>2 · Dim · FfnMult</c> (gate‖up).</summary>
    public int FfnMult { get; init; } = 4;

    public float Eps { get; init; } = 1e-5f;

    public double RopeTheta { get; init; } = 100.0;

    /// <summary>Fraction of each head rotated; 0.75 of 64 = 48 dims over 3 axes.</summary>
    public float RopeDimRatio { get; init; } = 0.75f;

    /// <summary>Decoder input channels (<c>z_channels</c>), what <c>post_quant_conv</c> emits.</summary>
    public int ZChannels { get; init; } = 24;

    /// <summary>Latent channels (<c>embed_dim</c>), what <c>post_quant_conv</c> consumes.</summary>
    public int LatentChannels { get; init; } = 24;

    public int OutChannels { get; init; } = 3;

    /// <summary>Pixels per latent cell on H/W; also the ViT patch size.</summary>
    public int VaeRatio { get; init; } = 16;

    /// <summary>Frames per latent token; also the ViT temporal patch size.</summary>
    public int VaeRatioT { get; init; } = 4;

    /// <summary>Frames the encoder consumed per clip; drives the whole temporal-chunk schedule.</summary>
    public int ClipLength { get; init; } = 17;

    /// <summary>Latent tokens the encoder dropped per clip; a non-zero value makes decode emit an overlap chunk.</summary>
    public int TokenDrop { get; init; } = 3;

    /// <summary>Spatial tile edge in PIXELS (tiles are cut in pixel space, then divided by <see cref="VaeRatio"/>).</summary>
    public int TileSize { get; init; } = 256;

    public int TileOverlapMin { get; init; } = 64;

    public bool DecoderTiling { get; init; } = true;

    /// <summary>Per-channel latent mean; decode denormalizes with <c>latent · std + mean</c> before the ViT.</summary>
    public float[] LatentsMean { get; init; } = DefaultLatentsMean;

    /// <summary>Per-channel latent standard deviation.</summary>
    public float[] LatentsStd { get; init; } = DefaultLatentsStd;

    /// <summary>Pixel normalization the encoder applied (ImageNet); decode folds its inverse into <c>proj_out</c>.</summary>
    public float[] PixelMean { get; init; } = [0.485f, 0.456f, 0.406f];

    public float[] PixelStd { get; init; } = [0.229f, 0.224f, 0.225f];

    /// <summary>Encoder stem width; each stage scales it by <see cref="EncoderChannelMult"/>.</summary>
    public int EncoderBaseChannels { get; init; } = 128;

    /// <summary>Per-stage channel multiplier of the 3D CNN encoder.</summary>
    public int[] EncoderChannelMult { get; init; } = [1, 2, 2, 4, 4, 8];

    /// <summary>Per-stage spatial stride; the product is <see cref="VaeRatio"/>. Strides are invisible in the
    /// checkpoint (a downsample weight is <c>[C,C,3,3,3]</c> whatever it strides by), so they come from config or here.</summary>
    public int[] EncoderSpaceDown { get; init; } = [2, 2, 2, 2, 1, 1];

    /// <summary>Per-stage temporal stride; the product is <see cref="VaeRatioT"/>.</summary>
    public int[] EncoderTimeDown { get; init; } = [1, 2, 2, 1, 1, 1];

    /// <summary>Residual blocks per encoder stage.</summary>
    public int EncoderResBlocks { get; init; } = 2;

    /// <summary>Groups in the encoder's group norms, whose statistics are taken per frame rather than per clip.</summary>
    public int EncoderNormGroups { get; init; } = 32;

    public float EncoderNormEps { get; init; } = 1e-6f;

    /// <summary>Transformer width.</summary>
    public int Dim => Heads * DimHead;

    /// <summary>Gated FFN inner width (half of what <c>w1</c> emits).</summary>
    public int FfnInner => Dim * FfnMult;

    /// <summary>Rotated head dims: 3 axes x <see cref="RopeInvFreqLen"/> freqs x 2 pair halves.</summary>
    public int RopeDim => (int)(DimHead * RopeDimRatio);

    /// <summary>Inverse frequencies per axis.</summary>
    public int RopeInvFreqLen => RopeDim / 6;

    /// <summary>Width of a <c>proj_out</c> row: one full pixel patch.</summary>
    public int PatchDim => OutChannels * VaeRatioT * VaeRatio * VaeRatio;

    /// <summary>Register tokens plus the zero cls row.</summary>
    public int NumSuffixTokens => NumRegisterTokens + 1;

    /// <summary>Leading decoded frames each chunk discards, from the clip length not being a multiple of the temporal ratio.</summary>
    public int FramePrePadding => Mod(-ClipLength, VaeRatioT);

    /// <summary>Latent tokens per temporal chunk.</summary>
    public int TokensChunkSize => (ClipLength + VaeRatioT - 1) / VaeRatioT;

    /// <summary>Extra tokens each chunk reads past its own span so the dropped tokens can be reconstructed.</summary>
    public int TokenOverlap => Mod(-TokenDrop, TokensChunkSize);

    /// <summary>Frames blended between consecutive temporal chunks.</summary>
    public int FrameOverlap => Math.Max(TokenOverlap * VaeRatioT - FramePrePadding, 0);

    /// <summary>Tile starts/lengths/overlaps in PIXEL space, shared by the encoder and decoder so both cut the same
    /// grid. Overlaps absorb the remainder in <see cref="VaeRatio"/> steps, keeping every boundary on a latent cell
    /// and guaranteeing that each tile advances by at least one latent unit.</summary>
    public (int[] Start, int[] Length, int[] Overlap) SplitTiles(int inputLen)
    {
        if (inputLen <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputLen), inputLen, "The tiled VAE axis must be positive.");
        }
        if (TileSize >= inputLen)
        {
            return ([0], [inputLen], []);
        }
        if (TileSize < VaeRatio * 2 || TileSize % VaeRatio != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(TileSize), TileSize,
                $"MiniMax-H3 VAE tile size must be a multiple of {VaeRatio} and at least {VaeRatio * 2} "
                + "when an axis needs multiple tiles.");
        }
        int overlapMin = Math.Clamp(TileOverlapMin, VaeRatio, TileSize - VaeRatio);
        overlapMin -= overlapMin % VaeRatio;
        int n = (inputLen + TileSize - 1) / TileSize;
        int[] overlaps;
        int remaining;
        while (true)
        {
            overlaps = new int[n - 1];
            Array.Fill(overlaps, overlapMin);
            int sum = 0;
            foreach (int o in overlaps)
            {
                sum += o;
            }
            remaining = TileSize * n - sum - inputLen;
            if (remaining < 0)
            {
                n++;
            }
            else
            {
                break;
            }
        }
        int units = remaining / VaeRatio;
        for (int i = 0; i < units; i++)
        {
            overlaps[i % (n - 1)] += VaeRatio;
        }
        int[] start = new int[n];
        for (int i = 1; i < n; i++)
        {
            start[i] = start[i - 1] + TileSize - overlaps[i - 1];
        }
        int[] length = new int[n];
        Array.Fill(length, TileSize);
        return (start, length, overlaps);
    }

    /// <summary>True when <paramref name="weights"/> carries the 3D CNN encoder signature; the encoder half ships in
    /// the same file as the decoder and is simply unread when only decoding.</summary>
    public static bool MatchesEncoder(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return weights.ContainsKey("encoder.conv_in.weight") && weights.ContainsKey("encoder.conv_out.weight")
            && weights.ContainsKey("quant_conv.weight");
    }

    /// <summary>True when <paramref name="weights"/> carries the ViT3D decoder signature.</summary>
    public static bool Matches(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return weights.ContainsKey("decoder.x_embedder.weight") && weights.ContainsKey("decoder.proj_out.weight")
            && weights.ContainsKey("decoder.transformer_blocks.0.attn.to_qkv.weight");
    }

    /// <summary>Recovers everything the checkpoint shapes expose (width, depth, channels, FFN multiplier, register
    /// token count, latent statistics) on top of <paramref name="baseline"/>, which supplies the head split and the
    /// rope/eps constants that no tensor shape encodes.</summary>
    public static MiniMaxH3VideoVaeConfig Detect(IReadOnlyDictionary<string, Tensor> weights,
        MiniMaxH3VideoVaeConfig? baseline = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!Matches(weights))
        {
            throw new ArgumentException(
                "Not a MiniMax-H3 video VAE: 'decoder.x_embedder.weight', 'decoder.proj_out.weight' and "
                + "'decoder.transformer_blocks.0.attn.to_qkv.weight' are all required.");
        }
        MiniMaxH3VideoVaeConfig baseConfig = baseline ?? new MiniMaxH3VideoVaeConfig();

        Tensor xEmbed = Require(weights, "decoder.x_embedder.weight");
        int dim = (int)xEmbed.Shape[0];
        int zChannels = (int)xEmbed.Shape[1];
        int ffnMult = (int)(Require(weights, "decoder.transformer_blocks.0.ff.w1.weight").Shape[0] / (2L * dim));
        int registerTokens = weights.TryGetValue("decoder.register_tokens", out Tensor? reg) ? (int)reg.Shape[1] : 0;
        int latentChannels = weights.TryGetValue("post_quant_conv.weight", out Tensor? pq) ? (int)pq.Shape[1]
            : baseConfig.LatentChannels;

        if (dim % baseConfig.DimHead != 0)
        {
            throw new ArgumentException(
                $"MiniMax-H3 video VAE width {dim} is not a multiple of the configured head dim {baseConfig.DimHead}.");
        }

        MiniMaxH3VideoVaeConfig config = baseConfig with
        {
            Heads = dim / baseConfig.DimHead,
            NumLayers = CountBlocks(weights, "decoder.transformer_blocks."),
            NumRegisterTokens = registerTokens,
            FfnMult = ffnMult,
            ZChannels = zChannels,
            LatentChannels = latentChannels,
            LatentsMean = ReadVector(weights, "latents_mean") ?? baseConfig.LatentsMean,
            LatentsStd = ReadVector(weights, "latents_std") ?? baseConfig.LatentsStd,
        };

        long patchDim = Require(weights, "decoder.proj_out.weight").Shape[0];
        if (patchDim != config.PatchDim)
        {
            throw new ArgumentException(
                $"MiniMax-H3 video VAE proj_out emits {patchDim} features but the configured patch "
                + $"({config.OutChannels}x{config.VaeRatioT}x{config.VaeRatio}x{config.VaeRatio}) needs {config.PatchDim}.");
        }
        return config;
    }

    /// <summary>Builds a config from the component <c>config.json</c> (the <c>vae_*</c> wrapper keys plus
    /// <c>latents_mean</c>/<c>latents_std</c>) and optionally the inner <c>source/config.json</c>, whose
    /// <c>vit_decoder_kwargs</c> is the only place the head split and rope constants are written down.</summary>
    public static MiniMaxH3VideoVaeConfig FromJson(string wrapperJson, string? sourceJson = null)
    {
        ArgumentNullException.ThrowIfNull(wrapperJson);
        MiniMaxH3VideoVaeConfig config = new MiniMaxH3VideoVaeConfig();

        using (JsonDocument doc = JsonDocument.Parse(wrapperJson))
        {
            JsonElement root = doc.RootElement;
            config = config with
            {
                LatentChannels = Int(root, "latent_channels", config.LatentChannels),
                ClipLength = Int(root, "vae_clip_length", config.ClipLength),
                TokenDrop = Int(root, "vae_token_drop", config.TokenDrop),
                TileSize = Int(root, "vae_tile_size", config.TileSize),
                TileOverlapMin = Int(root, "vae_tile_overlap_min", config.TileOverlapMin),
                DecoderTiling = Int(root, "vae_decoder_tiling", config.DecoderTiling ? 1 : 0) != 0,
                LatentsMean = Vector(root, "latents_mean") ?? config.LatentsMean,
                LatentsStd = Vector(root, "latents_std") ?? config.LatentsStd,
            };
        }

        if (sourceJson is null)
        {
            return config;
        }

        using JsonDocument source = JsonDocument.Parse(sourceJson);
        JsonElement s = source.RootElement;
        config = config with
        {
            ZChannels = Int(s, "z_channels", config.ZChannels),
            LatentChannels = Int(s, "embed_dim", config.LatentChannels),
            OutChannels = Int(s, "out_ch", config.OutChannels),
            VaeRatio = Product(s, "space_up", "space_down", config.VaeRatio),
            VaeRatioT = Product(s, "time_up", "time_down", config.VaeRatioT),
        };
        if (!s.TryGetProperty("vit_decoder_kwargs", out JsonElement vit) || vit.ValueKind != JsonValueKind.Object)
        {
            return config;
        }
        return config with
        {
            Heads = Int(vit, "heads", config.Heads),
            DimHead = Int(vit, "dim_head", config.DimHead),
            NumLayers = Int(vit, "num_layers", config.NumLayers),
            RopeTheta = Float(vit, "rope_theta", (float)config.RopeTheta),
            RopeDimRatio = Float(vit, "rope_dim_ratio", config.RopeDimRatio),
        };
    }

    private static int Mod(int value, int modulus) => ((value % modulus) + modulus) % modulus;

    private static int Int(JsonElement root, string key, int fallback) =>
        root.TryGetProperty(key, out JsonElement e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32() : fallback;

    private static float Float(JsonElement root, string key, float fallback) =>
        root.TryGetProperty(key, out JsonElement e) && e.ValueKind == JsonValueKind.Number ? (float)e.GetDouble() : fallback;

    private static float[]? Vector(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out JsonElement e) || e.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        float[] values = new float[e.GetArrayLength()];
        int i = 0;
        foreach (JsonElement item in e.EnumerateArray())
        {
            values[i++] = (float)item.GetDouble();
        }
        return values;
    }

    /// <summary>Product of the first present of <paramref name="primary"/>/<paramref name="fallbackKey"/> (the up
    /// list is null in the ViT-decoder configs, where the ratio must come from the encoder's down list).</summary>
    private static int Product(JsonElement root, string primary, string fallbackKey, int fallback)
    {
        foreach (string key in new[] { primary, fallbackKey })
        {
            if (!root.TryGetProperty(key, out JsonElement e) || e.ValueKind != JsonValueKind.Array)
            {
                continue;
            }
            int product = 1;
            foreach (JsonElement item in e.EnumerateArray())
            {
                product *= item.GetInt32();
            }
            return product;
        }
        return fallback;
    }

    private static unsafe float[]? ReadVector(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
        {
            return null;
        }
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        try
        {
            float[] values = new float[f32.ElementCount];
            new ReadOnlySpan<float>((float*)f32.DataPointer, values.Length).CopyTo(values);
            return values;
        }
        finally
        {
            if (!ReferenceEquals(f32, t)) f32.Dispose();
        }
    }

    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int count = 0;
        while (weights.ContainsKey($"{prefix}{count}.attn.to_qkv.weight"))
        {
            count++;
        }
        return count;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? t) ? t
            : throw new ArgumentException($"MiniMax-H3 video VAE checkpoint is missing '{key}'.");

    private static readonly float[] DefaultLatentsMean =
    [
        0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
        -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
        -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
        -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
        -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
        -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f,
    ];

    private static readonly float[] DefaultLatentsStd =
    [
        1.2223774194717407f, 1.2767263650894165f, 1.6831774711608886f, 1.7549455165863037f,
        1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
        0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
        0.7996809482574463f, 0.4498890042304992f, 0.7197399735450745f, 0.6936293244361877f,
        2.961095094680786f, 2.7694199085235595f, 3.0496184825897215f, 2.1088054180145265f,
        3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811525f,
    ];
}
