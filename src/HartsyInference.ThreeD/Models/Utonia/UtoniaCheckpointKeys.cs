using System.Globalization;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Maps Pointcept Utonia state-dict keys (PyTorch module paths) to the C# parameter names the encoder requests, and
/// enumerates the parameters a config implies. Names are <c>&lt;stage or block prefix&gt;.&lt;module&gt;.&lt;param&gt;</c>.
/// </summary>
public static class UtoniaCheckpointKeys
{
    public const string Weight = "Weight";
    public const string Bias = "Bias";
    public const string Gamma = "Gamma";

    public const string EmbeddingLinear = "Embedding.Linear";
    public const string EmbeddingNorm = "Embedding.Norm";
    public const string DownProj = "Down.Proj";
    public const string DownNorm = "Down.Norm";
    public const string UpLinear = "Up.Proj.Linear";
    public const string UpNorm = "Up.Proj.Norm";
    public const string SkipLinear = "Up.Skip.Linear";
    public const string SkipNorm = "Up.Skip.Norm";
    public const string CpeConv = "Cpe.Conv";
    public const string CpeLinear = "Cpe.Linear";
    public const string CpeNorm = "Cpe.Norm";
    public const string Norm1 = "Norm1";
    public const string Norm2 = "Norm2";
    public const string LayerScale1 = "LayerScale1";
    public const string LayerScale2 = "LayerScale2";
    public const string AttnQkv = "Attn.Qkv";
    public const string AttnProj = "Attn.Proj";
    public const string MlpFc1 = "Mlp.Fc1";
    public const string MlpFc2 = "Mlp.Fc2";

    /// <summary>Name prefix of an encoder or decoder stage.</summary>
    public static string StagePrefix(bool decoder, int stage) => (decoder ? "Decoder" : "Encoder") + ".Stage" + stage;

    /// <summary>Name prefix of one block inside a stage.</summary>
    public static string BlockPrefix(bool decoder, int stage, int block) => StagePrefix(decoder, stage) + ".Block" + block;

    /// <summary>Joins a prefix, a module and a parameter; an empty prefix is omitted.</summary>
    public static string ParameterName(string prefix, string module, string parameter) =>
        string.IsNullOrEmpty(prefix) ? module + "." + parameter : prefix + "." + module + "." + parameter;

    /// <summary>
    /// Maps one PyTorch state-dict key to its C# parameter name. Returns null for keys the encoder does not use,
    /// including computed buffers (see <see cref="IsComputedBuffer"/>).
    /// </summary>
    public static string? ToCSharpName(string torchKey)
    {
        int dot = torchKey.LastIndexOf('.');
        if (dot <= 0 || dot == torchKey.Length - 1)
        {
            return null;
        }

        string? module = ModuleName(torchKey[..dot]);
        string? parameter = torchKey[(dot + 1)..] switch
        {
            "weight" => Weight,
            "bias" => Bias,
            "gamma" => Gamma,
            _ => null
        };
        return module is null || parameter is null ? null : module + "." + parameter;
    }

    /// <summary>True for <c>attn.rope.inv_freq</c>, a buffer the encoder recomputes from the config instead of loading.</summary>
    public static bool IsComputedBuffer(string torchKey) => torchKey.EndsWith(".attn.rope.inv_freq", StringComparison.Ordinal);

    /// <summary>Every parameter the config implies, with its state-dict key, C# name and shape.</summary>
    public static IReadOnlyList<UtoniaParameterSpec> Parameters(UtoniaConfig config)
    {
        List<UtoniaParameterSpec> parameters = new();
        Walk(config, parameters, null);
        return parameters;
    }

    /// <summary>State-dict keys of computed buffers the config implies (one per block).</summary>
    public static IReadOnlyList<string> Buffers(UtoniaConfig config)
    {
        List<string> buffers = new();
        Walk(config, null, buffers);
        return buffers;
    }

    private static void Walk(UtoniaConfig config, List<UtoniaParameterSpec>? parameters, List<string>? buffers)
    {
        config.Validate();
        int stages = config.EncDepths.Length;
        int c0 = config.EncChannels[0];
        AddLinear(parameters, "embedding.stem.linear", EmbeddingLinear, c0, config.InChannels, true);
        AddNorm(parameters, "embedding.stem.norm", EmbeddingNorm, c0);
        for (int s = 0; s < stages; s++)
        {
            int c = config.EncChannels[s];
            if (s > 0)
            {
                string down = StagePrefix(false, s) + ".";
                AddLinear(parameters, "enc.enc" + s + ".down.proj", down + DownProj, c, config.EncChannels[s - 1], true);
                AddNorm(parameters, "enc.enc" + s + ".down.norm.0", down + DownNorm, c);
            }

            for (int i = 0; i < config.EncDepths[s]; i++)
            {
                WalkBlock(config, parameters, buffers, "enc.enc" + s + ".block" + i, BlockPrefix(false, s, i), c);
            }
        }

        if (config.EncoderOnly)
        {
            return;
        }

        int[] decExt = [.. config.DecChannels, config.EncChannels[stages - 1]];
        for (int s = 0; s < stages - 1; s++)
        {
            int d = config.DecChannels[s];
            string up = StagePrefix(true, s) + ".";
            string torchUp = "dec.dec" + s + ".up";
            AddLinear(parameters, torchUp + ".proj.0", up + UpLinear, d, decExt[s + 1], true);
            AddNorm(parameters, torchUp + ".proj.1", up + UpNorm, d);
            AddLinear(parameters, torchUp + ".proj_skip.0", up + SkipLinear, d, config.EncChannels[s], true);
            AddNorm(parameters, torchUp + ".proj_skip.1", up + SkipNorm, d);
            for (int i = 0; i < config.DecDepths[s]; i++)
            {
                WalkBlock(config, parameters, buffers, "dec.dec" + s + ".block" + i, BlockPrefix(true, s, i), d);
            }
        }
    }

    private static void WalkBlock(UtoniaConfig config, List<UtoniaParameterSpec>? parameters, List<string>? buffers,
        string torchBlock, string csBlock, int channels)
    {
        long c = channels;
        long hidden = (long)(channels * config.MlpRatio);
        string cs = csBlock + ".";
        if (parameters is not null)
        {
            parameters.Add(new UtoniaParameterSpec(torchBlock + ".cpe.0.weight", cs + CpeConv + ".Weight", [c, 3, 3, 3, c]));
            parameters.Add(new UtoniaParameterSpec(torchBlock + ".cpe.0.bias", cs + CpeConv + ".Bias", [c]));
        }

        AddLinear(parameters, torchBlock + ".cpe.1", cs + CpeLinear, c, c, true);
        AddNorm(parameters, torchBlock + ".cpe.2", cs + CpeNorm, c);
        AddNorm(parameters, torchBlock + ".norm1.0", cs + Norm1, c);
        if (config.LayerScaleInit is not null)
        {
            AddGamma(parameters, torchBlock + ".ls1.0", cs + LayerScale1, c);
        }

        AddLinear(parameters, torchBlock + ".attn.qkv", cs + AttnQkv, 3 * c, c, config.QkvBias);
        AddLinear(parameters, torchBlock + ".attn.proj", cs + AttnProj, c, c, true);
        buffers?.Add(torchBlock + ".attn.rope.inv_freq");
        AddNorm(parameters, torchBlock + ".norm2.0", cs + Norm2, c);
        if (config.LayerScaleInit is not null)
        {
            AddGamma(parameters, torchBlock + ".ls2.0", cs + LayerScale2, c);
        }

        AddLinear(parameters, torchBlock + ".mlp.0.fc1", cs + MlpFc1, hidden, c, true);
        AddLinear(parameters, torchBlock + ".mlp.0.fc2", cs + MlpFc2, c, hidden, true);
    }

    private static void AddLinear(List<UtoniaParameterSpec>? list, string torchModule, string csModule, long outDim, long inDim, bool bias)
    {
        if (list is null)
        {
            return;
        }

        list.Add(new UtoniaParameterSpec(torchModule + ".weight", csModule + "." + Weight, [outDim, inDim]));
        if (bias)
        {
            list.Add(new UtoniaParameterSpec(torchModule + ".bias", csModule + "." + Bias, [outDim]));
        }
    }

    private static void AddNorm(List<UtoniaParameterSpec>? list, string torchModule, string csModule, long dim)
    {
        if (list is null)
        {
            return;
        }

        list.Add(new UtoniaParameterSpec(torchModule + ".weight", csModule + "." + Weight, [dim]));
        list.Add(new UtoniaParameterSpec(torchModule + ".bias", csModule + "." + Bias, [dim]));
    }

    private static void AddGamma(List<UtoniaParameterSpec>? list, string torchModule, string csModule, long dim)
    {
        list?.Add(new UtoniaParameterSpec(torchModule + ".gamma", csModule + "." + Gamma, [dim]));
    }

    private static string? ModuleName(string path)
    {
        if (path == "embedding.stem.linear")
        {
            return EmbeddingLinear;
        }

        if (path == "embedding.stem.norm")
        {
            return EmbeddingNorm;
        }

        string[] parts = path.Split('.');
        if (parts.Length < 3)
        {
            return null;
        }

        bool decoder = parts[0] == "dec";
        if (!decoder && parts[0] != "enc")
        {
            return null;
        }

        if (!TryParseIndex(parts[1], decoder ? "dec" : "enc", out int stage))
        {
            return null;
        }

        string prefix = StagePrefix(decoder, stage) + ".";
        string rest = string.Join('.', parts, 2, parts.Length - 2);
        switch (rest)
        {
            case "down.proj" when !decoder:
                return prefix + DownProj;
            case "down.norm.0" when !decoder:
                return prefix + DownNorm;
            case "up.proj.0" when decoder:
                return prefix + UpLinear;
            case "up.proj.1" when decoder:
                return prefix + UpNorm;
            case "up.proj_skip.0" when decoder:
                return prefix + SkipLinear;
            case "up.proj_skip.1" when decoder:
                return prefix + SkipNorm;
        }

        if (parts[2].Length > 5 && parts[2].StartsWith("block", StringComparison.Ordinal)
            && TryParseIndex(parts[2], "block", out int block))
        {
            string? suffix = BlockSuffix(string.Join('.', parts, 3, parts.Length - 3));
            return suffix is null ? null : BlockPrefix(decoder, stage, block) + "." + suffix;
        }

        return null;
    }

    private static string? BlockSuffix(string inner) => inner switch
    {
        "cpe.0" => CpeConv,
        "cpe.1" => CpeLinear,
        "cpe.2" => CpeNorm,
        "norm1.0" => Norm1,
        "norm2.0" => Norm2,
        "ls1.0" => LayerScale1,
        "ls2.0" => LayerScale2,
        "attn.qkv" => AttnQkv,
        "attn.proj" => AttnProj,
        "mlp.0.fc1" => MlpFc1,
        "mlp.0.fc2" => MlpFc2,
        _ => null
    };

    private static bool TryParseIndex(string token, string prefix, out int value)
    {
        value = 0;
        if (!token.StartsWith(prefix, StringComparison.Ordinal))
        {
            return false;
        }

        return int.TryParse(token.AsSpan(prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out value);
    }
}
