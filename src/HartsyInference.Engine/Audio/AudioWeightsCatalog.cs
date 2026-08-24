using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>The ACE-Step 1.5 and YuE weight tables, ported from the extension's audio weights registry onto the Engine's <see cref="ModelAsset"/> + <see cref="ModelDownloader"/> machinery (per-target lock, SHA-256 verify, atomic move). Every ACE-Step variant is a DISTINCT checkpoint with its own verified hash — never alias them.</summary>
public static class AudioWeightsCatalog
{
    /// <summary>Catalog id of the ACE-Step music family.</summary>
    public const string AceStepId = "acestep";

    /// <summary>Catalog id of the YuE music family.</summary>
    public const string YueId = "yue";

    /// <summary>Models-root-relative folder ACE-Step checkpoints land in.</summary>
    internal const string AceStepSubdir = "audio/music/acestep";

    /// <summary>Models-root-relative folder YuE checkpoints land in.</summary>
    internal const string YueSubdir = "audio/music/yue";

    private const string SilenceLatentSha = "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b";

    /// <summary>The shared 600 s @ 25 Hz silence latent (byte-identical in every ACE-Step 1.5 repo, so one copy).</summary>
    private static readonly ModelAsset AceSilenceLatent = new ModelAsset
    {
        Repo = "ACE-Step/acestep-v15-base",
        RepoPath = "silence_latent.pt",
        TargetSubdir = AceStepSubdir,
        TargetName = "acestep-v15-silence_latent.pt",
        Role = "silence-latent",
        Sha256 = SilenceLatentSha,
    };

    /// <summary>Alternate sources, keyed by the primary asset's save name: tried only when the primary download fails, so an install prefers a small pre-converted repack but still succeeds off the canonical source.</summary>
    private static readonly Dictionary<string, ModelAsset> _fallbacks = new(StringComparer.Ordinal)
    {
        ["xcodec.safetensors"] = new ModelAsset
        {
            Repo = "m-a-p/xcodec_mini_infer",
            RepoPath = "final_ckpt/ckpt_00360000.pth",
            TargetSubdir = YueSubdir,
            TargetName = "xcodec/ckpt_00360000.pth",
            Role = "codec",
        },
    };

    /// <summary>The shared X-Codec acoustic decoder: a pre-converted decode-only repack (~0.8 GB), falling back to the canonical torch checkpoint which the YuE loader converts once on first load.</summary>
    private static readonly ModelAsset YueXCodec = new ModelAsset
    {
        Repo = "HartsyAI/YuE-xcodec-mini-safetensors",
        RepoPath = "xcodec.safetensors",
        TargetSubdir = YueSubdir,
        TargetName = "xcodec.safetensors",
        Role = "codec",
    };

    // Lazy: YueVariant() below reads _yueSharedQuality, which is declared further down this file. Static field
    // initializers run in textual order, so building this dictionary eagerly would NRE on _yueSharedQuality
    // before it's assigned. Deferring the build until first access (after the whole static ctor has run)
    // sidesteps the ordering hazard for good instead of just for today's field layout.
    private static readonly Lazy<Dictionary<string, Dictionary<string, ModelAsset[]>>> _registry = new(BuildRegistry);

    private static Dictionary<string, Dictionary<string, ModelAsset[]>> BuildRegistry() => new(StringComparer.OrdinalIgnoreCase)
    {
        [YueId] = new Dictionary<string, ModelAsset[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["en-cot"] = YueVariant("en-cot"),
            ["en-icl"] = YueVariant("en-icl"),
            ["zh-cot"] = YueVariant("zh-cot"),
            ["zh-icl"] = YueVariant("zh-icl"),
        },
        [AceStepId] = new Dictionary<string, ModelAsset[]>(StringComparer.OrdinalIgnoreCase)
        {
            // The default "turbo" checkpoint lives inside the Ace-Step1.5 bundle repo's subfolder.
            ["turbo"] = AceStep15Variant("Ace-Step1.5", "turbo",
                "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7", subdir: "acestep-v15-turbo/"),
            ["turbo-shift1"] = AceStep15Variant("acestep-v15-turbo-shift1", "turbo-shift1",
                "6a2ae0d66c957eb659fdb438f05e5d2ee62604c311f05ce7464203136a767dc3"),
            ["turbo-shift3"] = AceStep15Variant("acestep-v15-turbo-shift3", "turbo-shift3",
                "ce57ceb4a82890c87d3d6071c70b2392db91f6c7dcf3f96b32ec7205f0d40457"),
            ["turbo-continuous"] = AceStep15Variant("acestep-v15-turbo-continuous", "turbo-continuous",
                "cffa3e3a44d29cfc686b35d45aab3e4f9dc72c959e5e741d92a1dfc14ac35cd0"),
            ["sft"] = AceStep15Variant("acestep-v15-sft", "sft",
                "d4dd3a93870f06720027965b90771f529ab02094b3d29e2518f1d5e097e1af7e"),
            ["base"] = AceStep15Variant("acestep-v15-base", "base",
                "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b"),
            ["xl-turbo"] = AceStep15XlVariant("xl-turbo",
                "86a1afb0a1f711f0e3304ff65d874df3ae6783db683dcf982513fb9b6d14ae71"),
            ["xl-sft"] = AceStep15XlVariant("xl-sft",
                "3c05ae268353b3540fb1fd7db4fd77ffbda9802ec641b624e15648e030ecf3ce"),
            ["xl-base"] = AceStep15XlVariant("xl-base",
                "56bf816fc9a69a5f45635e867b2ad742e1e648eb51fadb7d124cb8332d2e0940"),
        },
    };

    /// <summary>Whether the family's checkpoint is a multi-file FOLDER (sharded weights + sidecars) rather than a single safetensors — its load path resolves to the variant directory.</summary>
    public static bool IsFolderCheckpoint(string familyId) =>
        string.Equals(familyId, YueId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Resolves the effective variant for a family: a bare model spec ("-m yue", no colon) makes <see cref="AudioModelSelector.Parse"/> hand the whole token through as the variant, so the literal family id (and the empty string) must map to the family's default registered variant — otherwise a bare id resolves to a nonexistent "{family}/{family}" checkpoint path (the 2026-07-24 audio sweep's YuE failure; same trap FxCatalog documents for bare "demucs").</summary>
    public static string NormalizeVariant(string familyId, string? variant)
    {
        string value = (variant ?? string.Empty).Trim();
        if (value.Length > 0 && !value.Equals(familyId, StringComparison.OrdinalIgnoreCase))
        {
            return value;
        }
        return string.Equals(familyId, YueId, StringComparison.OrdinalIgnoreCase) ? "en-cot"
            : string.Equals(familyId, AceStepId, StringComparison.OrdinalIgnoreCase) ? "turbo"
            : value;
    }

    /// <summary>Every file a specific variant needs; empty when the Engine has no entry for it.</summary>
    public static IReadOnlyList<ModelAsset> AssetsFor(string familyId, string variant)
    {
        if (familyId is null || variant is null)
        {
            return [];
        }
        return _registry.Value.TryGetValue(familyId, out Dictionary<string, ModelAsset[]>? variants)
            && variants.TryGetValue(variant.Trim(), out ModelAsset[]? assets) ? assets : [];
    }

    /// <summary>The PRIMARY checkpoint asset (the weights file) for a variant, or null when unregistered.</summary>
    internal static ModelAsset? Primary(string familyId, string variant)
    {
        IReadOnlyList<ModelAsset> assets = AssetsFor(familyId, variant);
        return assets.Count > 0 ? assets[0] : null;
    }

    /// <summary>Downloads the variant's whole file set, honoring the per-asset alternate source. Already-present files are skipped by the downloader's own presence check.</summary>
    internal static async Task EnsureAsync(string familyId, string variant, CancellationToken cancel)
    {
        variant = NormalizeVariant(familyId, variant);
        foreach (ModelAsset asset in AssetsFor(familyId, variant))
        {
            cancel.ThrowIfCancellationRequested();
            try
            {
                await ModelDownloader.EnsureSideModelAsync(asset, null, cancel).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                if (!cancel.IsCancellationRequested && _fallbacks.TryGetValue(asset.FileName, out ModelAsset? fallback))
                {
                    Logs.Warning($"[Audio] Primary source for '{asset.FileName}' failed ({ex.Message}); using the fallback source.");
                    await ModelDownloader.EnsureSideModelAsync(fallback, null, cancel).ConfigureAwait(false);
                    continue;
                }
                Logs.Error($"[Audio] Failed to fetch '{asset.FileName}' for {familyId}/{variant}: {ex.Message}", ex);
                throw;
            }
        }
    }

    /// <summary>The 3-file set for a single-file 2B ACE-Step variant: sha-pinned weights + the variant's config.json (small, non-LFS, unhashed) + the shared silence latent.</summary>
    private static ModelAsset[] AceStep15Variant(string repo, string variant, string weightsSha, string subdir = "") =>
    [
        new ModelAsset
        {
            Repo = $"ACE-Step/{repo}",
            RepoPath = $"{subdir}model.safetensors",
            TargetSubdir = AceStepSubdir,
            TargetName = $"acestep-v15-{variant}.safetensors",
            Role = "transformer",
            Sha256 = weightsSha,
        },
        new ModelAsset
        {
            Repo = $"ACE-Step/{repo}",
            RepoPath = $"{subdir}config.json",
            TargetSubdir = AceStepSubdir,
            TargetName = $"acestep-v15-{variant}.config.json",
            Role = "config",
        },
        AceSilenceLatent,
    ];

    /// <summary>XL variants (4B DiT, ~10 GB): weights are Comfy-Org's single-file bf16 merge (the official repos ship 4x5 GB shards); config.json is the official repo's (it carries the encoder_* dims the Engine needs).</summary>
    private static ModelAsset[] AceStep15XlVariant(string variant, string weightsSha) =>
    [
        new ModelAsset
        {
            Repo = "Comfy-Org/ace_step_1.5_ComfyUI_files",
            RepoPath = $"split_files/diffusion_models/acestep_v1.5_{variant.Replace('-', '_')}_bf16.safetensors",
            TargetSubdir = AceStepSubdir,
            TargetName = $"acestep-v15-{variant}.safetensors",
            Role = "transformer",
            Sha256 = weightsSha,
        },
        new ModelAsset
        {
            Repo = $"ACE-Step/acestep-v15-{variant}",
            RepoPath = "config.json",
            TargetSubdir = AceStepSubdir,
            TargetName = $"acestep-v15-{variant}.config.json",
            Role = "config",
        },
        AceSilenceLatent,
    ];

    /// <summary>YuE Stage-1 (7B, ~12.5 GB): the m-a-p folder — 3 sharded safetensors + index + configs + tokenizer, landing in a per-variant subfolder — plus the shared X-Codec decoder.</summary>
    private static ModelAsset[] YueVariant(string variant)
    {
        string[] files =
        [
            "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
            "model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.model",
        ];
        List<ModelAsset> assets = new(files.Length + 1 + _yueSharedQuality.Length);
        for (int i = 0; i < files.Length; i++)
        {
            assets.Add(new ModelAsset
            {
                Repo = $"m-a-p/YuE-s1-7B-anneal-{variant}",
                RepoPath = files[i],
                TargetSubdir = YueSubdir,
                TargetName = $"{variant}/{files[i]}",
                Role = i == 0 ? "transformer" : "component",
            });
        }
        assets.Add(YueXCodec);
        assets.AddRange(_yueSharedQuality);
        return [.. assets];
    }

    // Stage-2 residual upsampler (cb0 → all 8 codebooks) + the per-stem Vocos vocoder torch checkpoints
    // (YueMusicModel auto-converts the .pth pair to safetensors on first load). Without these, YuE degrades
    // to the vocal-cb0-only 16 kHz x-codec draft — the "garbled" mode. Shared across variants (parent-level
    // siblings of the variant folder).
    private static readonly ModelAsset[] _yueSharedQuality =
    [
        new ModelAsset
        {
            Repo = "m-a-p/YuE-s2-1B-general",
            RepoPath = "model.safetensors",
            TargetSubdir = YueSubdir,
            TargetName = "s2/model.safetensors",
            Role = "component",
        },
        new ModelAsset
        {
            Repo = "m-a-p/YuE-s2-1B-general",
            RepoPath = "config.json",
            TargetSubdir = YueSubdir,
            TargetName = "s2/config.json",
            Role = "component",
        },
        new ModelAsset
        {
            Repo = "m-a-p/xcodec_mini_infer",
            RepoPath = "decoders/decoder_131000.pth",
            TargetSubdir = YueSubdir,
            TargetName = "xcodec_vocoder_src/decoder_131000.pth",
            Role = "component",
        },
        new ModelAsset
        {
            Repo = "m-a-p/xcodec_mini_infer",
            RepoPath = "decoders/decoder_151000.pth",
            TargetSubdir = YueSubdir,
            TargetName = "xcodec_vocoder_src/decoder_151000.pth",
            Role = "component",
        },
    ];
}
