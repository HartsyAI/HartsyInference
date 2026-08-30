using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine.Planning;

/// <summary>Resolves one header-first, hash-bound video plan without materializing checkpoint tensor payloads.</summary>
internal static class VideoProfileResolver
{
    private const string H3FamilyId = "minimax-h3";
    private const string H3LicenseUrl = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE";
    private const string H3FunControlSha256 =
        "919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02";
    private const string H3FunPrunedFormat = "minimax_h3_fun_pruned_v1";

    /// <summary>Builds a plan for a resolved recipe family and request.</summary>
    public static async Task<VideoPlan> ResolveAsync(ModelSpec spec, VideoRequest request, string familyId,
        VideoDefaults familyDefaults, VideoFeatures familyFeatures, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        ArgumentNullException.ThrowIfNull(familyDefaults);

        if (!string.Equals(familyId, H3FamilyId, StringComparison.OrdinalIgnoreCase))
        {
            return ResolveGeneric(spec, request, familyId, familyDefaults, familyFeatures);
        }
        return await ResolveH3Async(spec, request, familyDefaults, familyFeatures, cancel).ConfigureAwait(false);
    }

    private static VideoPlan ResolveGeneric(ModelSpec spec, VideoRequest request, string familyId,
        VideoDefaults defaults, VideoFeatures features)
    {
        List<VideoPlanIssue> issues = [];
        ValidateCommonRequest(request, issues);
        VideoRequest resolved = defaults.Apply(request);
        VideoFeatures requested = VideoService.RequestedFeatures(request);
        VideoFeatures missing = requested & ~features;
        if (missing != VideoFeatures.None)
        {
            issues.Add(Error("video.feature.unsupported",
                $"Video model family '{familyId}' does not support: {missing}.", "request"));
        }
        if (request.SparseAttentionPolicy == SparseAttentionPolicy.Require)
        {
            issues.Add(Error("video.vsa.profile_required",
                $"Video model family '{familyId}' has no validated sparse-attention profile.",
                nameof(VideoRequest.SparseAttentionPolicy)));
        }
        VideoModelProfile profile = new VideoModelProfile
        {
            Id = $"{familyId}-default",
            DisplayName = $"{familyId} default",
            FamilyId = familyId,
            Task = VideoTaskFamily.Unknown,
            Acceleration = VideoAccelerationKind.None,
            Attention = VideoAttentionKind.Dense,
            Defaults = defaults,
            Features = features,
        };
        if (!string.IsNullOrWhiteSpace(spec.ProfileId)
            && !string.Equals(spec.ProfileId, profile.Id, StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.profile.unknown",
                $"Profile '{spec.ProfileId}' is not registered for video family '{familyId}'.", nameof(ModelSpec.ProfileId)));
        }
        VideoEffectiveSettings settings = EffectiveFromResolved(resolved, defaults.LockedFields);
        return new VideoPlan
        {
            SourceRequest = request,
            Model = spec,
            Profile = profile,
            EffectiveSettings = settings,
            Issues = issues,
            CacheIdentity = $"profile:{profile.Id};",
            ComponentPaths = spec.LocalPath is null
                ? new Dictionary<string, string>(StringComparer.Ordinal)
                : new Dictionary<string, string>(StringComparer.Ordinal) { ["checkpoint"] = spec.LocalPath },
        };
    }

    private static async Task<VideoPlan> ResolveH3Async(ModelSpec spec, VideoRequest request, VideoDefaults familyDefaults,
        VideoFeatures familyFeatures, CancellationToken cancel)
    {
        List<VideoPlanIssue> issues = [];
        ValidateCommonRequest(request, issues, deferSourceFreeMaskVideo: true);
        ValidateH3Request(request, issues);
        await ValidateSourceFreeMaskVideoAsync(request.VideoDenoiseMask, issues, cancel).ConfigureAwait(false);
        Dictionary<string, string> componentPaths = new Dictionary<string, string>(StringComparer.Ordinal);
        Dictionary<string, string> componentFormats = new Dictionary<string, string>(StringComparer.Ordinal);
        Dictionary<string, string> artifactHashes = new Dictionary<string, string>(StringComparer.Ordinal);
        Dictionary<string, IReadOnlyDictionary<string, string>> artifactMetadata =
            new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal);

        if (string.IsNullOrWhiteSpace(spec.LocalPath))
        {
            issues.Add(Error("video.model.missing", "MiniMax-H3 planning requires a resolved local checkpoint path.",
                nameof(ModelSpec.LocalPath)));
            return InvalidH3Plan(spec, request, familyDefaults, familyFeatures, issues);
        }

        string checkpointPath = spec.LocalPath;
        MiniMaxH3Assets? assets = null;
        try
        {
            assets = MiniMaxH3Assets.Resolve(checkpointPath, request.Components);
            checkpointPath = assets.Dit;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            issues.Add(Error("video.component.missing", ex.Message, nameof(ModelSpec.LocalPath)));
            if (!File.Exists(checkpointPath))
            {
                return InvalidH3Plan(spec, request, familyDefaults, familyFeatures, issues);
            }
        }

        HeaderSnapshot header;
        try
        {
            header = ReadHeader(checkpointPath);
        }
        catch (Exception ex) when (ex is IOException or JsonException or InvalidOperationException
            or HartsyInference.Core.Exceptions.HartsyInferenceException)
        {
            issues.Add(Error("video.checkpoint.header_invalid",
                $"Could not inspect MiniMax-H3 checkpoint header: {ex.Message}", nameof(ModelSpec.LocalPath)));
            return InvalidH3Plan(spec, request, familyDefaults, familyFeatures, issues);
        }

        ValidateH3Structure(header.Descriptors, issues);
        string quantization = DetectFormat(header.Descriptors);
        componentPaths["transformer"] = checkpointPath;
        componentFormats["transformer"] = quantization;
        artifactMetadata["transformer"] = header.Metadata;

        // A safetensors header can prove that the payload is loadable, but community repacks commonly preserve
        // tensor names while changing task or acceleration semantics. Only an exact byte identity may select those.
        string mainHash;
        try
        {
            mainHash = await VideoCheckpointHashCache.GetSha256Async(checkpointPath, cancel).ConfigureAwait(false);
            artifactHashes["transformer"] = mainHash;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            issues.Add(Error("video.checkpoint.hash_failed",
                $"Could not hash MiniMax-H3 checkpoint: {ex.Message}", nameof(ModelSpec.LocalPath)));
            return InvalidH3Plan(spec, request, familyDefaults, familyFeatures, issues);
        }

        VideoModelProfile profile;
        if (VideoProfileManifest.TryGetByHash(mainHash, out VideoKnownArtifact? known) && known is not null)
        {
            // The manifest deliberately indexes every artifact kind together so a user cannot point the primary
            // model field at a valid adapter/component hash and accidentally construct it under base semantics.
            if (known.Role == VideoProfileArtifactRole.Rejected)
            {
                issues.Add(Error("video.checkpoint.incompatible_layout", known.RejectionReason!, nameof(ModelSpec.LocalPath)));
                profile = UnknownBaseProfile(header.Metadata, quantization, mainHash, familyDefaults, familyFeatures);
            }
            else if (known.Role != VideoProfileArtifactRole.Main)
            {
                issues.Add(Error("video.checkpoint.wrong_artifact_role",
                    $"'{known.DisplayName}' is a {known.Role} artifact, not a primary MiniMax-H3 transformer.",
                    nameof(ModelSpec.LocalPath)));
                profile = UnknownBaseProfile(header.Metadata, quantization, mainHash, familyDefaults, familyFeatures);
            }
            else
            {
                profile = ProfileFromKnown(known, header.Metadata, quantization);
            }
        }
        else
        {
            // Unknown hashes stay on conservative base defaults unless a local sidecar binds the same exact bytes.
            // Filenames and checkpoint metadata are only hints because both survive many incompatible conversions.
            VideoProfileSidecar? sidecar = await ReadSidecarAsync(checkpointPath, mainHash, header.Descriptors,
                issues, cancel).ConfigureAwait(false);
            profile = sidecar is null
                ? UnknownBaseProfile(header.Metadata, quantization, mainHash, familyDefaults, familyFeatures)
                : ProfileFromSidecar(sidecar, header.Metadata, quantization, mainHash);
            AddFilenameHintIssues(checkpointPath, profile, sidecar is not null, issues);
        }
        ValidateVsaProfileStructure(header.Descriptors, profile, issues);

        await ResolveLoraCompositionAsync(request, profile, header.Descriptors, artifactHashes, componentPaths,
            componentFormats, issues, artifactMetadata, cancel,
            resolvedProfile => profile = resolvedProfile).ConfigureAwait(false);
        if (string.Equals(profile.Id, "minimax-h3-ref2va-zs05-int8", StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.profile.composition_required",
                "The Joey ZS05 checkpoint is certified only in the exact composition with the LightX FL8-768p adapter.",
                nameof(VideoRequest.Loras)));
        }
        await ResolveControlComponentsAsync(request, mainHash, header, profile, artifactHashes, componentPaths,
            componentFormats, artifactMetadata, issues, cancel).ConfigureAwait(false);

        if (assets is not null)
        {
            string videoVae = request.Components?.VideoVae ?? request.Components?.Vae ?? assets.VideoVae;
            string? audioVae = request.Components?.AudioVae ?? assets.AudioVae;
            await AddComponentAsync("videoVae", videoVae, hashForManifest: true, componentPaths, componentFormats,
                artifactHashes, issues, cancel).ConfigureAwait(false);
            if (audioVae is not null)
            {
                await AddComponentAsync("audioVae", audioVae, hashForManifest: false, componentPaths, componentFormats,
                    artifactHashes, issues, cancel).ConfigureAwait(false);
            }
            await AddComponentAsync("textEncoder", assets.TextEncoder, hashForManifest: false, componentPaths,
                componentFormats, artifactHashes, issues, cancel).ConfigureAwait(false);
        }

        ValidateProfileHint(spec.ProfileId, profile, mainHash, issues);
        ValidateRequestedFeatures(request, profile, issues);
        VideoEffectiveSettings settings = ResolveEffectiveSettings(request, profile, issues);
        issues.Add(Warning("minimax.h3.license",
            "MiniMax-H3 use is subject to its community license, including territory restrictions and notice, disclosure, and display obligations."));

        string cacheIdentity = BuildCacheIdentity(profile, artifactHashes, componentPaths);
        return new VideoPlan
        {
            SourceRequest = request,
            Model = spec,
            Profile = profile,
            EffectiveSettings = settings,
            Issues = issues,
            CacheIdentity = cacheIdentity,
            ComponentPaths = componentPaths,
            ComponentFormats = componentFormats,
            ArtifactHashes = artifactHashes,
            ArtifactMetadata = artifactMetadata,
        };
    }

    private static VideoPlan InvalidH3Plan(ModelSpec spec, VideoRequest request, VideoDefaults defaults,
        VideoFeatures features, List<VideoPlanIssue> issues)
    {
        VideoModelProfile profile = UnknownBaseProfile(new Dictionary<string, string>(StringComparer.Ordinal),
            "unknown", null, defaults, features);
        return new VideoPlan
        {
            SourceRequest = request,
            Model = spec,
            Profile = profile,
            EffectiveSettings = ResolveEffectiveSettings(request, profile, issues),
            Issues = issues,
            CacheIdentity = $"profile:{profile.Id};",
        };
    }

    private static VideoModelProfile UnknownBaseProfile(IReadOnlyDictionary<string, string> metadata,
        string quantization, string? hash, VideoDefaults familyDefaults, VideoFeatures familyFeatures) =>
        new VideoModelProfile
        {
            Id = "minimax-h3-unrecognized-base",
            DisplayName = "Unrecognized MiniMax-H3 base checkpoint",
            FamilyId = H3FamilyId,
            Task = VideoTaskFamily.Unknown,
            Acceleration = VideoAccelerationKind.None,
            Attention = VideoAttentionKind.Dense,
            Defaults = H3Defaults(familyDefaults.Steps, 12f, 3f, VideoLockedFields.CfgScale),
            Features = familyFeatures,
            ArtifactSha256 = hash,
            Quantization = quantization,
            CheckpointMetadata = metadata,
            LicenseUrl = H3LicenseUrl,
        };

    private static VideoModelProfile ProfileFromKnown(VideoKnownArtifact artifact,
        IReadOnlyDictionary<string, string> metadata, string quantization)
    {
        VideoLockedFields locked = VideoLockedFields.CfgScale;
        if (artifact.Acceleration is VideoAccelerationKind.Turbo or VideoAccelerationKind.Vsa)
        {
            locked |= VideoLockedFields.Steps | VideoLockedFields.FlowShift | VideoLockedFields.AudioFlowShift
                | VideoLockedFields.Sampler | VideoLockedFields.Scheduler;
        }
        if (artifact.Width is not null || artifact.Height is not null)
        {
            locked |= VideoLockedFields.Geometry;
        }
        if (artifact.Acceleration == VideoAccelerationKind.Pdd)
        {
            locked |= VideoLockedFields.FlowShift | VideoLockedFields.AudioFlowShift | VideoLockedFields.Sampler
                | VideoLockedFields.Scheduler;
        }
        VideoDefaults defaults = H3Defaults(artifact.Steps ?? 30, artifact.FlowShift ?? 12f,
            artifact.AudioFlowShift ?? 3f, locked, artifact.Width, artifact.Height,
            referenceSizing: artifact.ReferenceSizing);
        return new VideoModelProfile
        {
            Id = artifact.Id,
            DisplayName = artifact.DisplayName,
            FamilyId = H3FamilyId,
            Task = artifact.Task,
            Acceleration = artifact.Acceleration,
            Attention = artifact.Attention,
            Defaults = defaults,
            Features = artifact.Attention == VideoAttentionKind.Dense
                ? FeaturesForTask(artifact.Task) : VideoFeatures.Lora,
            ArtifactSha256 = artifact.Sha256,
            Quantization = quantization,
            IsBuiltIn = true,
            ProvenanceUrl = artifact.ProvenanceUrl,
            LicenseUrl = H3LicenseUrl,
            CheckpointMetadata = metadata,
        };
    }

    private static VideoModelProfile ProfileFromSidecar(VideoProfileSidecar sidecar,
        IReadOnlyDictionary<string, string> metadata, string quantization, string hash)
    {
        VideoLockedFields locked = VideoLockedFields.Steps | VideoLockedFields.CfgScale | VideoLockedFields.FlowShift
            | VideoLockedFields.AudioFlowShift | VideoLockedFields.Sampler | VideoLockedFields.Scheduler;
        if (sidecar.Width is not null || sidecar.Height is not null)
        {
            locked |= VideoLockedFields.Geometry;
        }
        VideoDefaults defaults = H3Defaults(sidecar.Steps, sidecar.FlowShift, sidecar.AudioFlowShift, locked,
            sidecar.Width, sidecar.Height, sidecar.CfgScale, sidecar.Sampler, sidecar.Scheduler,
            sidecar.ReferenceSizing);
        return new VideoModelProfile
        {
            Id = sidecar.ProfileId,
            DisplayName = sidecar.DisplayName ?? sidecar.ProfileId,
            FamilyId = H3FamilyId,
            Task = sidecar.Task,
            Acceleration = sidecar.Acceleration,
            Attention = sidecar.Attention,
            Defaults = defaults,
            Features = sidecar.Attention == VideoAttentionKind.Dense
                ? FeaturesForTask(sidecar.Task) : VideoFeatures.Lora,
            ArtifactSha256 = hash,
            Quantization = quantization,
            IsSidecar = true,
            ProvenanceUrl = sidecar.ProvenanceUrl,
            LicenseUrl = sidecar.LicenseUrl ?? H3LicenseUrl,
            CheckpointMetadata = metadata,
        };
    }

    private static VideoDefaults H3Defaults(int steps, float flowShift, float audioFlowShift,
        VideoLockedFields locked, int? width = null, int? height = null, float cfgScale = 1f,
        string sampler = "euler", string scheduler = "normal",
        VideoReferenceSizing referenceSizing = VideoReferenceSizing.Native) =>
        new VideoDefaults
        {
            Steps = steps,
            CfgScale = cfgScale,
            Width = width ?? 1344,
            Height = height ?? 768,
            Frames = 124,
            Fps = MiniMaxH3Geometry.Fps,
            FlowShift = flowShift,
            AudioFlowShift = audioFlowShift,
            Sampler = sampler,
            Scheduler = scheduler,
            LockedFields = locked,
            ReferenceSizing = referenceSizing,
        };

    private static VideoFeatures FeaturesForTask(VideoTaskFamily task) => task switch
    {
        VideoTaskFamily.T2Va => VideoFeatures.VideoDenoiseMask | VideoFeatures.AudioDenoiseMask | VideoFeatures.Lora,
        VideoTaskFamily.Fl2Va => VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.Guides
            | VideoFeatures.VideoDenoiseMask | VideoFeatures.AudioDenoiseMask | VideoFeatures.VideoControlNet
            | VideoFeatures.VideoInpaint | VideoFeatures.Lora,
        VideoTaskFamily.Ref2Va => VideoFeatures.ReferenceImages | VideoFeatures.ReferenceVideos
            | VideoFeatures.ReferenceAudios | VideoFeatures.VideoDenoiseMask
            | VideoFeatures.AudioDenoiseMask | VideoFeatures.Lora,
        VideoTaskFamily.Hybrid => VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.ReferenceImages
            | VideoFeatures.ReferenceVideos | VideoFeatures.ReferenceAudios | VideoFeatures.Guides
            | VideoFeatures.VideoDenoiseMask | VideoFeatures.AudioDenoiseMask | VideoFeatures.Lora,
        _ => VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.ReferenceImages
            | VideoFeatures.ReferenceVideos | VideoFeatures.ReferenceAudios | VideoFeatures.Guides
            | VideoFeatures.VideoDenoiseMask | VideoFeatures.AudioDenoiseMask | VideoFeatures.Lora,
    };

    private static async Task ResolveLoraCompositionAsync(VideoRequest request, VideoModelProfile initialProfile,
        IReadOnlyDictionary<string, SafeTensorDescriptor> baseDescriptors,
        Dictionary<string, string> artifactHashes, Dictionary<string, string> componentPaths,
        Dictionary<string, string> componentFormats, List<VideoPlanIssue> issues,
        Dictionary<string, IReadOnlyDictionary<string, string>> artifactMetadata, CancellationToken cancel,
        Action<VideoModelProfile> setProfile)
    {
        List<LoraResolver.LoraSpec> loras;
        try
        {
            loras = LoraResolver.Resolve(request.Loras);
        }
        catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException)
        {
            issues.Add(Error("video.lora.missing", ex.Message, nameof(VideoRequest.Loras)));
            return;
        }
        VideoModelProfile profile = initialProfile;
        VideoKnownArtifact? distillation = null;
        int distillationIndex = -1;
        bool distillationBindsUnknownTask = false;
        bool prunedBase = NormalizeDescriptors(baseDescriptors).ContainsKey("adaln_t_table");
        for (int i = 0; i < loras.Count; i++)
        {
            LoraResolver.LoraSpec lora = loras[i];
            string hash;
            try
            {
                hash = await VideoCheckpointHashCache.GetSha256Async(lora.FilePath, cancel).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                issues.Add(Error("video.lora.hash_failed", $"Could not hash LoRA '{lora.FilePath}': {ex.Message}",
                    nameof(VideoRequest.Loras)));
                continue;
            }
            string role = $"lora:{i}";
            artifactHashes[role] = hash;
            componentPaths[role] = lora.FilePath;
            HeaderSnapshot? loraHeader = TryReadComponentHeader(lora.FilePath, issues, role);
            componentFormats[role] = loraHeader is null ? "invalid" : DetectFormat(loraHeader.Descriptors);
            if (loraHeader is not null)
            {
                artifactMetadata[role] = loraHeader.Metadata;
            }
            bool isPddHeader = loraHeader is not null && MiniMaxH3PddAdapter.IsPddHeader(loraHeader.Descriptors);
            if (loraHeader is not null && !isPddHeader)
            {
                ValidateOrdinaryLoraCompatibility(loraHeader.Descriptors, baseDescriptors, role, issues);
            }
            // Acceleration artifacts use a LoRA-shaped container, but they also replace the step schedule and output
            // heads. Exact role lookup separates those contracts from an ordinary merge before construction.
            VideoKnownArtifact? artifact = null;
            bool known = VideoProfileManifest.TryGetByHash(hash, out artifact) && artifact is not null;
            if (!known && isPddHeader && loraHeader is not null)
            {
                // A local PDD rebase necessarily has a new full-file hash. Its converter metadata is accepted only
                // because it records the official provenance and binds the exact selected pruned base.
                artifact = ConvertedPddArtifact(hash, loraHeader.Metadata, profile.ArtifactSha256, issues);
            }
            if (artifact is null || artifact.Role != VideoProfileArtifactRole.Adapter)
            {
                if (isPddHeader)
                {
                    issues.Add(Error("video.pdd.profile_required",
                        $"PDD adapter '{Path.GetFileName(lora.FilePath)}' is not a known official hash or a "
                        + "Hartsy converter output bound to this base checkpoint.", nameof(VideoRequest.Loras)));
                }
                continue;
            }
            if (artifact.Acceleration == VideoAccelerationKind.Pdd && !isPddHeader)
            {
                issues.Add(Error("video.pdd.banks_missing",
                    $"Known PDD adapter '{artifact.DisplayName}' does not expose all four projection banks.",
                    nameof(VideoRequest.Loras)));
                continue;
            }
            bool convertedPdd = artifact.Acceleration == VideoAccelerationKind.Pdd && !known;
            int validationIssueCount = issues.Count;
            if (artifact.Acceleration == VideoAccelerationKind.Pdd && loraHeader is not null)
            {
                ValidatePddHeader(loraHeader, artifact, issues);
            }
            ValidateAdapterBaseBinding(artifact, convertedPdd, prunedBase, profile.Task, issues);
            if (issues.Skip(validationIssueCount).Any(issue => issue.Severity == VideoPlanIssueSeverity.Error))
            {
                continue;
            }
            if (distillation is not null)
            {
                // Two acceleration adapters would each claim the one global sampler schedule and projection-head
                // bank; choosing either by stack order would make the request's execution contract ambiguous.
                issues.Add(Error("video.acceleration.multiple",
                    $"Distillation adapters '{distillation.DisplayName}' and '{artifact.DisplayName}' cannot be stacked.",
                    nameof(VideoRequest.Loras)));
                continue;
            }
            distillation = artifact;
            distillationIndex = i;
            distillationBindsUnknownTask = convertedPdd;
            if (!float.IsFinite(lora.ModelStrength) || !float.IsFinite(lora.TencStrength)
                || Math.Abs(lora.ModelStrength - 1f) > 1e-6f || Math.Abs(lora.TencStrength - 1f) > 1e-6f)
            {
                issues.Add(Error("video.acceleration.strength_locked",
                    $"'{artifact.DisplayName}' is an acceleration profile and requires model/text strengths of exactly 1.",
                    nameof(VideoRequest.Loras)));
            }
        }
        if (distillation is null)
        {
            return;
        }
        if (profile.Acceleration != VideoAccelerationKind.None)
        {
            issues.Add(Error("video.acceleration.incompatible",
                $"Profile '{profile.DisplayName}' already bakes {profile.Acceleration} and cannot stack '{distillation.DisplayName}'.",
                nameof(VideoRequest.Loras)));
            return;
        }

        bool joeyComposition = string.Equals(profile.Id, "minimax-h3-ref2va-zs05-int8", StringComparison.OrdinalIgnoreCase)
            && string.Equals(distillation.Id, "minimax-h3-lightx-fl8-768p", StringComparison.OrdinalIgnoreCase);
        // Joey ZS05 was validated only as this cross-task pair, so the composed profile is an explicit manifest
        // exception rather than a general license to attach FL adapters to Ref2VA bases.
        if (!joeyComposition && profile.Task == VideoTaskFamily.Unknown && !distillationBindsUnknownTask)
        {
            issues.Add(Error("video.profile.task_unbound",
                $"'{distillation.DisplayName}' targets {distillation.Task}, but the unrecognized base has no "
                + "hash- or sidecar-bound task. Add a matching hash-bound video profile sidecar.",
                nameof(VideoRequest.Loras)));
            return;
        }
        if (!joeyComposition && profile.Task != VideoTaskFamily.Unknown && profile.Task != distillation.Task)
        {
            issues.Add(Error("video.profile.task_mismatch",
                $"'{distillation.DisplayName}' targets {distillation.Task}, but the base checkpoint is {profile.Task}.",
                nameof(VideoRequest.Loras)));
            return;
        }
        if (string.Equals(profile.Id, "minimax-h3-ref2va-zs05-int8", StringComparison.OrdinalIgnoreCase)
            && !joeyComposition)
        {
            issues.Add(Error("video.profile.composition_unverified",
                "The Joey ZS05 base is certified only with the LightX FL8-768p adapter.", nameof(VideoRequest.Loras)));
            return;
        }

        VideoKnownArtifact effectiveArtifact = distillation with
        {
            Id = joeyComposition ? "minimax-h3-ref2va-zs05-lightx-fl8-768p" : distillation.Id,
            DisplayName = joeyComposition ? "Joey ZS05 Ref2VA + LightX FL8-768p" : distillation.DisplayName,
            Task = joeyComposition ? VideoTaskFamily.Ref2Va : distillation.Task,
        };
        VideoModelProfile composed = ProfileFromKnown(effectiveArtifact, profile.CheckpointMetadata,
            profile.Quantization ?? "unknown") with
        {
            ArtifactSha256 = profile.ArtifactSha256,
            ProvenanceUrl = effectiveArtifact.ProvenanceUrl,
        };
        artifactHashes[$"profileAdapter:{distillationIndex}"] = distillation.Sha256;
        setProfile(composed);
    }

    private static void ValidateAdapterBaseBinding(VideoKnownArtifact artifact, bool convertedPdd,
        bool prunedBase, VideoTaskFamily baseTask, List<VideoPlanIssue> issues)
    {
        if (artifact.Acceleration != VideoAccelerationKind.Pdd)
        {
            return;
        }
        if (convertedPdd && !prunedBase)
        {
            issues.Add(Error("video.pdd.converted_base_not_pruned",
                "A locally rebased PDD adapter is certified only for its pruned adaln_t_table target base.",
                nameof(VideoRequest.Loras)));
        }
        else if (!convertedPdd && prunedBase)
        {
            issues.Add(Error("video.pdd.rebase_required",
                "Official PDD adapters target a full-width H3 base. Run 'hartsy convert h3-pdd' before using one "
                + "with a pruned adaln_t_table checkpoint.", nameof(VideoRequest.Loras)));
        }
        if (!convertedPdd && baseTask == VideoTaskFamily.Unknown)
        {
            issues.Add(Error("video.profile.task_unbound",
                $"Official {artifact.Task} PDD cannot attach to a base whose task is not hash- or sidecar-bound.",
                nameof(VideoRequest.Loras)));
        }
    }

    private static void ValidateProfileHint(string? profileId, VideoModelProfile profile, string mainHash,
        List<VideoPlanIssue> issues)
    {
        if (string.IsNullOrWhiteSpace(profileId) || string.Equals(profileId, profile.Id, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }
        if (VideoProfileManifest.TryGetById(profileId, out VideoKnownArtifact? hinted) && hinted is not null)
        {
            if (hinted.Role == VideoProfileArtifactRole.Main && !string.Equals(hinted.Sha256, mainHash,
                StringComparison.OrdinalIgnoreCase))
            {
                issues.Add(Error("video.profile.hash_conflict",
                    $"Profile '{profileId}' is bound to SHA-256 {hinted.Sha256}, but the selected checkpoint hashes to {mainHash}.",
                    nameof(ModelSpec.ProfileId)));
                return;
            }
        }
        issues.Add(Error("video.profile.mismatch",
            $"Requested profile '{profileId}' does not match detected profile '{profile.Id}'.", nameof(ModelSpec.ProfileId)));
    }

    private static void ValidateRequestedFeatures(VideoRequest request, VideoModelProfile profile,
        List<VideoPlanIssue> issues)
    {
        VideoFeatures requested = VideoService.RequestedFeatures(request);
        VideoFeatures missing = requested & ~profile.Features;
        if (missing != VideoFeatures.None)
        {
            issues.Add(Error("video.profile.feature_incompatible",
                $"Profile '{profile.Id}' does not support: {missing}.", "request"));
        }
        bool hasReference = request.ReferenceImages is { Count: > 0 } || request.ReferenceVideos is { Count: > 0 }
            || request.ReferenceAudios is { Count: > 0 };
        if (profile.Task == VideoTaskFamily.Ref2Va && !hasReference)
        {
            issues.Add(Error("video.profile.reference_required",
                $"Profile '{profile.Id}' requires at least one reference image, video, or audio clip.", "reference"));
        }
        if (profile.Attention != VideoAttentionKind.Dense && requested != VideoFeatures.None)
        {
            VideoFeatures forbidden = requested & ~VideoFeatures.Lora;
            if (forbidden != VideoFeatures.None)
            {
                issues.Add(Error("video.vsa.conditioning_incompatible",
                    $"Sparse profile '{profile.Id}' is T2VA-only and cannot use: {forbidden}.", "request"));
            }
        }
        if (profile.Attention != VideoAttentionKind.Dense && request.SparseAttentionPolicy == SparseAttentionPolicy.Disable)
        {
            issues.Add(Error("video.vsa.disable_unsupported",
                $"Profile '{profile.Id}' requires sparse attention and does not certify dense equivalence.",
                nameof(VideoRequest.SparseAttentionPolicy)));
        }
        if (profile.Attention == VideoAttentionKind.Dense && request.SparseAttentionPolicy == SparseAttentionPolicy.Require)
        {
            issues.Add(Error("video.vsa.profile_required",
                $"Profile '{profile.Id}' carries no validated VSA gates.", nameof(VideoRequest.SparseAttentionPolicy)));
        }

        if (profile.Acceleration == VideoAccelerationKind.Pdd
            && (request.VideoDenoiseMask is not null || request.AudioDenoiseMask is not null))
        {
            issues.Add(Error("video.pdd.mask_incompatible",
                "PDD evaluates only its trained global sigma knots and cannot use per-row video or audio mask timesteps.",
                request.VideoDenoiseMask is not null
                    ? nameof(VideoRequest.VideoDenoiseMask) : nameof(VideoRequest.AudioDenoiseMask)));
        }

        bool hasControls = request.Controls is { Count: > 0 };
        if (hasControls && (profile.Acceleration != VideoAccelerationKind.None || profile.Task == VideoTaskFamily.Hybrid))
        {
            issues.Add(Error("video.control.acceleration_incompatible",
                "Fun ControlNet cannot initially combine with Turbo, PDD, VSA, or Hybrid execution.",
                nameof(VideoRequest.Controls)));
        }
        if (hasControls && profile.Task != VideoTaskFamily.Fl2Va)
        {
            issues.Add(Error("video.control.base_incompatible",
                "Fun ControlNet requires a hash- or sidecar-bound FL2VA base profile.",
                nameof(VideoRequest.Controls)));
        }
        bool hasInpaint = request.Controls?.Any(control => control.Kind == VideoControlKind.Inpaint) == true;
        if (hasInpaint && (request.VideoDenoiseMask is not null || request.AudioDenoiseMask is not null))
        {
            issues.Add(Error("video.control.mask_precedence_ambiguous",
                "ControlNet inpainting and sampler denoise masks cannot run together.", nameof(VideoRequest.Controls)));
        }
    }

    private static void ValidateCommonRequest(VideoRequest request, List<VideoPlanIssue> issues,
        bool deferSourceFreeMaskVideo = false)
    {
        if (request.Guides is { Count: > 0 })
        {
            HashSet<int> visualFrames = [];
            HashSet<int> audioFrames = [];
            for (int i = 0; i < request.Guides.Count; i++)
            {
                VideoGuide guide = request.Guides[i];
                bool hasImage = guide.Image is not null;
                bool hasVideo = guide.Video is not null;
                if (hasImage == hasVideo && (hasImage || guide.Audio is null))
                {
                    issues.Add(Error("video.guide.visual_xor",
                        $"Guide {i} must set image xor video, with audio optionally paired; an audio-only guide may set neither.",
                        nameof(VideoRequest.Guides)));
                }
                if ((hasImage || hasVideo) && !visualFrames.Add(guide.FrameIndex))
                {
                    issues.Add(Error("video.guide.duplicate_visual",
                        $"More than one visual guide targets frame {guide.FrameIndex}.", nameof(VideoRequest.Guides)));
                }
                if (guide.Audio is not null && !audioFrames.Add(guide.FrameIndex))
                {
                    issues.Add(Error("video.guide.duplicate_audio",
                        $"More than one audio guide targets frame {guide.FrameIndex}.", nameof(VideoRequest.Guides)));
                }
                if (guide.Image is not null)
                {
                    ValidateImage(guide.Image, $"guide {i} image", nameof(VideoRequest.Guides), issues);
                }
                if (guide.Video is not null && guide.Video.Data.Length == 0)
                {
                    issues.Add(Error("video.guide.video_empty", $"Guide {i} video has no encoded bytes.",
                        nameof(VideoRequest.Guides)));
                }
            }
        }

        if (request.VideoDenoiseMask is VideoDenoiseMask videoMask)
        {
            bool hasMaskImage = videoMask.MaskImage is not null;
            bool hasMaskVideo = videoMask.MaskVideo is not null;
            if (hasMaskImage == hasMaskVideo)
            {
                issues.Add(Error("video.mask.payload_xor", "VideoDenoiseMask must set maskImage xor maskVideo.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
            bool hasSourceImage = videoMask.SourceImage is not null;
            bool hasSourceVideo = videoMask.SourceVideo is not null;
            if (hasSourceImage && hasSourceVideo)
            {
                issues.Add(Error("video.mask.source_xor", "VideoDenoiseMask may set sourceImage xor sourceVideo, not both.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
            bool allWhiteImage = videoMask.MaskImage is not null
                && videoMask.MaskImage.Rgb.Length > 0 && videoMask.MaskImage.Rgb.All(value => value == byte.MaxValue);
            bool sourceFreeMaskVideoPending = deferSourceFreeMaskVideo && videoMask.MaskVideo is not null;
            if (!allWhiteImage && !sourceFreeMaskVideoPending && !hasSourceImage && !hasSourceVideo)
            {
                issues.Add(Error("video.mask.source_required",
                    "A video mask that may preserve rows requires an explicit source; only a provably all-white image mask is source-free.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
            if (videoMask.MaskImage is not null)
            {
                ValidateImage(videoMask.MaskImage, "video mask", nameof(VideoRequest.VideoDenoiseMask), issues);
            }
            if (videoMask.SourceImage is not null)
            {
                ValidateImage(videoMask.SourceImage, "video mask source", nameof(VideoRequest.VideoDenoiseMask), issues);
            }
            if (videoMask.MaskVideo is not null && videoMask.MaskVideo.Data.Length == 0)
            {
                issues.Add(Error("video.mask.video_empty", "Video mask clip has no encoded bytes.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
            if (videoMask.SourceVideo is not null && videoMask.SourceVideo.Data.Length == 0)
            {
                issues.Add(Error("video.mask.source_empty", "Video mask source clip has no encoded bytes.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
        }

        if (request.AudioDenoiseMask is AudioDenoiseMask audioMask)
        {
            if (audioMask.Values.Count == 0)
            {
                issues.Add(Error("audio.mask.empty", "Audio denoise mask needs at least one value.",
                    nameof(VideoRequest.AudioDenoiseMask)));
            }
            if (!float.IsFinite(audioMask.Rate) || audioMask.Rate <= 0f)
            {
                issues.Add(Error("audio.mask.rate_invalid", "Audio mask rate must be finite and greater than zero.",
                    nameof(VideoRequest.AudioDenoiseMask)));
            }
            bool preserves = false;
            for (int i = 0; i < audioMask.Values.Count; i++)
            {
                float value = audioMask.Values[i];
                if (!UnitInterval.Contains(value))
                {
                    issues.Add(Error("audio.mask.value_invalid",
                        $"Audio mask value {i} must be finite and within [0,1].", nameof(VideoRequest.AudioDenoiseMask)));
                    break;
                }
                preserves |= value < 1f;
            }
            if (preserves && audioMask.Source is null)
            {
                issues.Add(Error("audio.mask.source_required",
                    "An audio mask with any value below one requires source audio.", nameof(VideoRequest.AudioDenoiseMask)));
            }
        }

        if (request.Controls is { Count: > 0 })
        {
            for (int i = 0; i < request.Controls.Count; i++)
            {
                VideoControl control = request.Controls[i];
                if (string.IsNullOrWhiteSpace(control.Model))
                {
                    issues.Add(Error("video.control.model_missing", $"Control {i} has no model path.",
                        nameof(VideoRequest.Controls)));
                }
                if (control.Video.Data.Length == 0)
                {
                    issues.Add(Error("video.control.video_empty", $"Control {i} has no encoded video bytes.",
                        nameof(VideoRequest.Controls)));
                }
                if (!VideoControlValidation.IsValidStrength(control.Strength))
                {
                    issues.Add(Error("video.control.strength_invalid",
                        $"Control {i} strength must be finite, non-negative, and representable as F32.",
                        nameof(VideoRequest.Controls)));
                }
                if (!VideoControlValidation.IsValidWindow(control.Start, control.End))
                {
                    issues.Add(Error("video.control.window_invalid",
                        $"Control {i} start/end must satisfy 0 <= start <= end <= 1.", nameof(VideoRequest.Controls)));
                }
                bool inpaint = control.Kind == VideoControlKind.Inpaint;
                if (inpaint && (control.VisibilityMask is null || control.MaskedSource is null))
                {
                    issues.Add(Error("video.control.inpaint_payload_missing",
                        $"Inpaint control {i} requires visibilityMask and maskedSource videos.", nameof(VideoRequest.Controls)));
                }
                if (!inpaint && (control.VisibilityMask is not null || control.MaskedSource is not null))
                {
                    issues.Add(Error("video.control.inpaint_payload_unexpected",
                        $"Control {i} supplies inpaint payloads but its kind is {control.Kind}.", nameof(VideoRequest.Controls)));
                }
                if (control.VisibilityMask is not null && control.VisibilityMask.Data.Length == 0)
                {
                    issues.Add(Error("video.control.visibility_empty", $"Control {i} visibility mask is empty.",
                        nameof(VideoRequest.Controls)));
                }
                if (control.MaskedSource is not null && control.MaskedSource.Data.Length == 0)
                {
                    issues.Add(Error("video.control.masked_source_empty", $"Control {i} masked source is empty.",
                        nameof(VideoRequest.Controls)));
                }
            }
        }
    }

    private static void ValidateH3Request(VideoRequest request, List<VideoPlanIssue> issues)
    {
        ValidatePositive(request.Width, nameof(VideoRequest.Width), "width", issues,
            maximum: MiniMaxH3Geometry.MaxPixels / MiniMaxH3Geometry.CanvasMultiple);
        ValidatePositive(request.Height, nameof(VideoRequest.Height), "height", issues,
            maximum: MiniMaxH3Geometry.MaxPixels / MiniMaxH3Geometry.CanvasMultiple);
        ValidatePositive(request.Frames, nameof(VideoRequest.Frames), "frame count", issues,
            maximum: int.MaxValue - 16);
        ValidatePositive(request.Fps, nameof(VideoRequest.Fps), "frame rate", issues);
        ValidatePositive(request.Steps, nameof(VideoRequest.Steps), "step count", issues);

        ValidateFinitePositive(request.CfgScale, nameof(VideoRequest.CfgScale), "CFG scale", issues);
        ValidateFinitePositive(request.FlowShift, nameof(VideoRequest.FlowShift), "video flow shift", issues);
        ValidateFinitePositive(request.AudioFlowShift, nameof(VideoRequest.AudioFlowShift), "audio flow shift", issues);

        if (request.Sampler is not null
            && !string.Equals(request.Sampler, "euler", StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.sampler.unsupported",
                "MiniMax-H3 supports only the Euler sampler.", nameof(VideoRequest.Sampler)));
        }
        if (request.Scheduler is not null
            && !string.Equals(request.Scheduler, "normal", StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.scheduler.unsupported",
                "MiniMax-H3 supports only the normal scheduler.", nameof(VideoRequest.Scheduler)));
        }
        if (request.ReferenceSizing is VideoReferenceSizing sizing && !Enum.IsDefined(sizing))
        {
            issues.Add(Error("video.reference_sizing.invalid",
                $"Unknown MiniMax-H3 reference-sizing value {(int)sizing}.", nameof(VideoRequest.ReferenceSizing)));
        }
        if (!Enum.IsDefined(request.SparseAttentionPolicy))
        {
            issues.Add(Error("video.vsa.policy_invalid",
                $"Unknown sparse-attention policy value {(int)request.SparseAttentionPolicy}.",
                nameof(VideoRequest.SparseAttentionPolicy)));
        }
        if (request.TrimVideoStartFrames < 0 || request.TrimVideoEndFrames < 0)
        {
            issues.Add(Error("video.trim.invalid", "Video trim counts must be non-negative.", "trim"));
        }
        if (request.Loras?.Entries is { } loras)
        {
            for (int i = 0; i < loras.Count; i++)
            {
                LoraEntry entry = loras[i];
                if (string.IsNullOrWhiteSpace(entry.Model))
                {
                    issues.Add(Error("video.lora.model_missing", $"LoRA {i} has no model path.",
                        nameof(VideoRequest.Loras)));
                }
                if (!double.IsFinite(entry.Weight)
                    || entry.TextEncoderWeight is double textWeight && !double.IsFinite(textWeight))
                {
                    issues.Add(Error("video.lora.strength_invalid",
                        $"LoRA {i} model/text strengths must be finite.", nameof(VideoRequest.Loras)));
                }
            }
        }
    }

    private static void ValidatePositive(int? value, string field, string description, List<VideoPlanIssue> issues,
        int maximum = int.MaxValue)
    {
        if (value is not int resolved)
        {
            return;
        }
        if (resolved <= 0 || resolved > maximum)
        {
            issues.Add(Error("video.request.value_invalid",
                $"MiniMax-H3 {description} must be in [1,{maximum}]; got {resolved}.", field));
        }
    }

    private static void ValidateFinitePositive(float? value, string field, string description,
        List<VideoPlanIssue> issues)
    {
        if (value is float resolved && (!float.IsFinite(resolved) || resolved <= 0f))
        {
            issues.Add(Error("video.request.value_invalid",
                $"MiniMax-H3 {description} must be finite and greater than zero; got "
                + resolved.ToString("R", CultureInfo.InvariantCulture) + ".", field));
        }
    }

    private static async Task ValidateSourceFreeMaskVideoAsync(VideoDenoiseMask? mask,
        List<VideoPlanIssue> issues, CancellationToken cancel)
    {
        if (mask?.MaskVideo is not VideoClip clip || mask.SourceImage is not null || mask.SourceVideo is not null
            || clip.Data.Length == 0)
        {
            return;
        }
        try
        {
            FfmpegProcessDecoder.Result decoded = await new FfmpegProcessDecoder()
                .DecodeAsync(clip.Data, clip.Format, cancel).ConfigureAwait(false);
            bool allWhite = decoded.Frames.Count > 0
                && decoded.Frames.All(frame => frame.Length > 0 && frame.All(value => value == byte.MaxValue));
            decoded.Frames.Clear();
            if (!allWhite)
            {
                issues.Add(Error("video.mask.source_required",
                    "A source-free video mask is accepted only when every decoded RGB value is exactly white; "
                    + "this clip may preserve rows and therefore requires sourceImage or sourceVideo.",
                    nameof(VideoRequest.VideoDenoiseMask)));
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException
            or ArgumentException or FormatException or OverflowException)
        {
            issues.Add(Error("video.mask.decode_invalid",
                $"Could not prove the source-free video mask is all-white: {ex.Message}",
                nameof(VideoRequest.VideoDenoiseMask)));
        }
    }

    private static void ValidateImage(ImageData image, string description, string field, List<VideoPlanIssue> issues)
    {
        if (image.Width <= 0 || image.Height <= 0 || image.Rgb.Length != (long)image.Width * image.Height * 3L)
        {
            issues.Add(Error("video.image.payload_invalid",
                $"{description} must contain width*height*3 RGB24 bytes for positive dimensions.", field));
        }
    }

    private static void ValidateResolvedGuideFrames(VideoRequest request, int frameCount,
        List<VideoPlanIssue> issues)
    {
        IReadOnlyList<VideoGuide>? guides = request.Guides;
        if (guides is null || guides.Count == 0)
        {
            return;
        }
        HashSet<int> visual = [];
        HashSet<int> audio = [];
        if (request.InitImage is not null)
        {
            visual.Add(0);
        }
        if (request.VideoEndFrame is not null)
        {
            visual.Add(frameCount - 1);
        }
        for (int i = 0; i < guides.Count; i++)
        {
            VideoGuide guide = guides[i];
            int resolved = guide.FrameIndex < 0 ? frameCount + guide.FrameIndex : guide.FrameIndex;
            if (resolved < 0 || resolved >= frameCount)
            {
                issues.Add(Error("video.guide.frame_out_of_range",
                    $"Guide {i} frame {guide.FrameIndex} resolves outside the aligned {frameCount}-frame target.",
                    nameof(VideoRequest.Guides)));
                continue;
            }
            if ((guide.Image is not null || guide.Video is not null) && !visual.Add(resolved))
            {
                issues.Add(Error("video.guide.duplicate_visual",
                    $"Multiple visual guides resolve to target frame {resolved}.", nameof(VideoRequest.Guides)));
            }
            if (guide.Audio is not null && !audio.Add(resolved))
            {
                issues.Add(Error("video.guide.duplicate_audio",
                    $"Multiple audio guides resolve to target frame {resolved}.", nameof(VideoRequest.Guides)));
            }
        }
    }

    private static async Task ResolveControlComponentsAsync(VideoRequest request, string baseHash,
        HeaderSnapshot baseHeader, VideoModelProfile profile, Dictionary<string, string> artifactHashes,
        Dictionary<string, string> componentPaths, Dictionary<string, string> componentFormats,
        Dictionary<string, IReadOnlyDictionary<string, string>> artifactMetadata,
        List<VideoPlanIssue> issues, CancellationToken cancel)
    {
        if (request.Controls is null || request.Controls.Count == 0)
        {
            return;
        }
        Dictionary<string, int> unique = new Dictionary<string, int>(VideoArtifactPath.Comparer);
        Dictionary<string, SafeTensorDescriptor> normalizedBase = NormalizeDescriptors(baseHeader.Descriptors);
        bool prunedBase = normalizedBase.ContainsKey("adaln_t_table");
        for (int i = 0; i < request.Controls.Count; i++)
        {
            string requestedPath = request.Controls[i].Model;
            if (string.IsNullOrWhiteSpace(requestedPath) || !File.Exists(requestedPath))
            {
                if (!string.IsNullOrWhiteSpace(requestedPath))
                {
                    issues.Add(Error("video.control.model_missing",
                        $"Control model '{requestedPath}' was not found locally.", nameof(VideoRequest.Controls)));
                }
                continue;
            }
            string path = VideoArtifactPath.Canonicalize(requestedPath);
            if (unique.ContainsKey(path))
            {
                continue;
            }
            int slot = unique.Count;
            unique[path] = slot;
            string role = $"controlModel:{slot}";
            componentPaths[role] = path;
            HeaderSnapshot? controlHeader = TryReadComponentHeader(path, issues, role);
            if (controlHeader is null)
            {
                componentFormats[role] = "invalid";
                continue;
            }
            artifactMetadata[role] = controlHeader.Metadata;
            string quantization = DetectFormat(controlHeader.Descriptors);
            int? controlTimeDim = ValidateFunControlHeader(controlHeader.Descriptors, issues, role);
            try
            {
                string hash = await VideoCheckpointHashCache.GetSha256Async(path, cancel).ConfigureAwait(false);
                artifactHashes[role] = hash;
                // The official branch targets full-width AdaLN. A pruned-base conversion gets a different identity,
                // so converter provenance plus an exact target-base hash is the only safe alternate admission path.
                bool official = string.Equals(hash, H3FunControlSha256, StringComparison.OrdinalIgnoreCase);
                bool rebased = IsRebasedFunControl(controlHeader.Metadata, baseHash, issues, role);
                componentFormats[role] = official
                    ? $"h3-fun-full-{quantization}"
                    : rebased ? $"h3-fun-pruned-rebased-{quantization}" : $"unrecognized-{quantization}";

                if (VideoProfileManifest.TryGetByHash(hash, out VideoKnownArtifact? artifact) && artifact is not null
                    && artifact.Role != VideoProfileArtifactRole.ControlNet)
                {
                    issues.Add(Error("video.control.wrong_artifact_role",
                        $"'{artifact.DisplayName}' cannot be used as a ControlNet branch.", nameof(VideoRequest.Controls)));
                }
                if (!official && !rebased)
                {
                    issues.Add(Error("video.control.unrecognized_artifact",
                        "Fun ControlNet accepts only the official branch hash or a local h3-controlnet conversion "
                        + "whose metadata binds the exact selected base.", nameof(VideoRequest.Controls)));
                }
                if (profile.Task != VideoTaskFamily.Fl2Va)
                {
                    issues.Add(Error("video.control.base_incompatible",
                        $"Fun ControlNet requires an FL2VA base, but profile '{profile.Id}' is {profile.Task}.",
                        nameof(VideoRequest.Controls)));
                }
                if (official && prunedBase)
                {
                    issues.Add(Error("video.control.rebase_required",
                        "The official Fun branch uses full-width AdaLN and cannot attach directly to this pruned "
                        + "base. Run the local h3-controlnet converter against this exact target base.",
                        nameof(VideoRequest.Controls)));
                }
                if (rebased && !prunedBase)
                {
                    issues.Add(Error("video.control.rebased_base_incompatible",
                        "A locally rebased Fun branch is valid only with its bound pruned base; use the official "
                        + "branch with a full-width base.", nameof(VideoRequest.Controls)));
                }
                int? expectedTimeDim = BaseTimeEmbedDim(normalizedBase);
                if (controlTimeDim is int actual && expectedTimeDim is int expected && actual != expected)
                {
                    issues.Add(Error("video.control.adaln_width_mismatch",
                        $"Fun control AdaLN width {actual} does not match the selected base width {expected}.",
                        nameof(VideoRequest.Controls)));
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                issues.Add(Error("video.control.hash_failed", $"Could not hash control model '{path}': {ex.Message}",
                    nameof(VideoRequest.Controls)));
            }
        }
    }

    private static bool IsRebasedFunControl(IReadOnlyDictionary<string, string> metadata, string baseHash,
        List<VideoPlanIssue> issues, string role)
    {
        if (!metadata.TryGetValue("hartsy.controlnet.format", out string? format))
        {
            return false;
        }
        if (!string.Equals(format, H3FunPrunedFormat, StringComparison.Ordinal))
        {
            issues.Add(Error("video.control.converted_format_invalid",
                $"Unknown converted Fun ControlNet format '{format}'.", role));
            return false;
        }

        bool valid = true;
        if (!metadata.TryGetValue("hartsy.controlnet.target_base_sha256", out string? targetHash)
            || !IsSha256(targetHash))
        {
            issues.Add(Error("video.control.target_hash_missing",
                "Converted Fun ControlNet metadata must record a valid target-base SHA-256.", role));
            valid = false;
        }
        else if (!string.Equals(targetHash, baseHash, StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.control.target_hash_mismatch",
                $"Converted Fun branch targets base SHA-256 {targetHash}, but the selected base is {baseHash}.", role));
            valid = false;
        }
        if (!metadata.TryGetValue("hartsy.controlnet.control_sha256", out string? controlHash)
            || !string.Equals(controlHash, H3FunControlSha256, StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.control.source_hash_invalid",
                "Converted Fun ControlNet metadata must bind the official source branch SHA-256.", role));
            valid = false;
        }
        if (!metadata.TryGetValue("hartsy.controlnet.full_base_sha256", out string? fullBaseHash)
            || !IsSha256(fullBaseHash))
        {
            issues.Add(Error("video.control.full_base_hash_missing",
                "Converted Fun ControlNet metadata must record a valid full-base SHA-256.", role));
            valid = false;
        }
        if (!metadata.TryGetValue("hartsy.controlnet.affine_residual", out string? residualText)
            || !double.TryParse(residualText, NumberStyles.Float, CultureInfo.InvariantCulture,
                out double residual) || !double.IsFinite(residual) || residual < 0.0 || residual > 1e-4)
        {
            issues.Add(Error("video.control.affine_residual_invalid",
                "Converted Fun ControlNet metadata must record an affine residual no greater than 1e-4.", role));
            valid = false;
        }
        return valid;
    }

    /// <summary>Header-only validation of the exact published five-block branch. Both VideoX-Fun split-QKV and
    /// Comfy fused-QKV layouts are accepted because the payload converter normalizes them losslessly.</summary>
    private static int? ValidateFunControlHeader(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        List<VideoPlanIssue> issues, string role)
    {
        Dictionary<string, SafeTensorDescriptor> normalized = NormalizeFunControlDescriptors(descriptors);
        ValidateControlShape(normalized, "control_proj_in.weight", [5376, 196], issues, role);
        ValidateControlShape(normalized, "control_proj_in.bias", [5376], issues, role);

        HashSet<int> blockIndices = new HashSet<int>();
        foreach (string key in normalized.Keys)
        {
            if (!key.StartsWith("control_blocks.", StringComparison.Ordinal))
            {
                continue;
            }
            int start = "control_blocks.".Length;
            int stop = key.IndexOf('.', start);
            if (stop > start && int.TryParse(key.AsSpan(start, stop - start), out int index))
            {
                blockIndices.Add(index);
            }
        }
        if (!blockIndices.SetEquals([0, 1, 2, 3, 4]))
        {
            issues.Add(Error("video.control.block_count_invalid",
                $"Fun ControlNet must contain exactly blocks 0..4; found [{string.Join(',', blockIndices.Order())}].",
                role));
        }

        int? timeDim = null;
        for (int index = 0; index < 5; index++)
        {
            string prefix = $"control_blocks.{index}";
            ValidateControlShape(normalized, prefix + ".norm1.weight", [5376], issues, role);
            ValidateControlShape(normalized, prefix + ".norm2.weight", [5376], issues, role);
            ValidateControlShape(normalized, prefix + ".attn.q_norm.weight", [128], issues, role);
            ValidateControlShape(normalized, prefix + ".attn.k_norm.weight", [128], issues, role);
            ValidateControlShape(normalized, prefix + ".attn.out_proj.weight", [5376, 7168], issues, role);
            ValidateControlShape(normalized, prefix + ".mlp.fc1.weight", [28672, 5376], issues, role);
            ValidateControlShape(normalized, prefix + ".mlp.fc2.weight", [5376, 14336], issues, role);
            ValidateControlShape(normalized, prefix + ".after_proj.weight", [5376, 5376], issues, role);
            ValidateControlShape(normalized, prefix + ".after_proj.bias", [5376], issues, role);
            if (index == 0)
            {
                ValidateControlShape(normalized, prefix + ".before_proj.weight", [5376, 5376], issues, role);
                ValidateControlShape(normalized, prefix + ".before_proj.bias", [5376], issues, role);
            }

            string adalnWeight = prefix + ".adaln_proj.linear.weight";
            if (!normalized.TryGetValue(adalnWeight, out SafeTensorDescriptor? adaln)
                || adaln.Shape.Rank != 2 || adaln.Shape[0] != 96768 || adaln.Shape[1] <= 0)
            {
                issues.Add(Error("video.control.tensor_shape_invalid",
                    $"Fun ControlNet tensor '{adalnWeight}' must be [96768,time], got "
                    + $"{(adaln is null ? "missing" : adaln.Shape)}.", role));
            }
            else if (timeDim is null)
            {
                timeDim = checked((int)adaln.Shape[1]);
            }
            else if (adaln.Shape[1] != timeDim.Value)
            {
                issues.Add(Error("video.control.adaln_width_inconsistent",
                    $"Fun control block {index} has AdaLN width {adaln.Shape[1]}, expected {timeDim.Value}.", role));
            }
            ValidateControlShape(normalized, prefix + ".adaln_proj.linear.bias", [96768], issues, role);

            string fused = prefix + ".attn.qkv_proj.weight";
            string q = prefix + ".attn.to_q.weight";
            string k = prefix + ".attn.to_k.weight";
            string v = prefix + ".attn.to_v.weight";
            bool hasFused = normalized.ContainsKey(fused);
            bool hasAnySplit = normalized.ContainsKey(q) || normalized.ContainsKey(k) || normalized.ContainsKey(v);
            if (hasFused == hasAnySplit)
            {
                issues.Add(Error("video.control.qkv_layout_invalid",
                    $"Fun control block {index} must contain fused QKV xor a complete split Q/K/V set.", role));
            }
            else if (hasFused)
            {
                ValidateControlShape(normalized, fused, [21504, 5376], issues, role);
            }
            else
            {
                ValidateControlShape(normalized, q, [7168, 5376], issues, role);
                ValidateControlShape(normalized, k, [7168, 5376], issues, role);
                ValidateControlShape(normalized, v, [7168, 5376], issues, role);
            }
        }
        return timeDim;
    }

    private static Dictionary<string, SafeTensorDescriptor> NormalizeFunControlDescriptors(
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        Dictionary<string, SafeTensorDescriptor> normalized =
            new Dictionary<string, SafeTensorDescriptor>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in descriptors)
        {
            string key = entry.Key;
            foreach (string prefix in new[] { "model.controlnet.", "controlnet.", "control_model." })
            {
                if (key.StartsWith(prefix, StringComparison.Ordinal))
                {
                    key = key[prefix.Length..];
                    break;
                }
            }
            key = key.Replace(".attn.norm_q.", ".attn.q_norm.", StringComparison.Ordinal)
                .Replace(".attn.norm_k.", ".attn.k_norm.", StringComparison.Ordinal)
                .Replace(".attn.to_out.0.", ".attn.out_proj.", StringComparison.Ordinal)
                .Replace(".ff.net.0.proj.", ".mlp.fc1.", StringComparison.Ordinal)
                .Replace(".ff.net.2.", ".mlp.fc2.", StringComparison.Ordinal);
            normalized[key] = entry.Value;
        }
        return normalized;
    }

    private static void ValidateControlShape(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string key, long[] expected, List<VideoPlanIssue> issues, string role)
    {
        if (!descriptors.TryGetValue(key, out SafeTensorDescriptor? descriptor)
            || descriptor.Shape.Rank != expected.Length
            || expected.Where((dimension, index) => descriptor.Shape[index] != dimension).Any())
        {
            issues.Add(Error("video.control.tensor_shape_invalid",
                $"Fun ControlNet tensor '{key}' must be [{string.Join(',', expected)}], got "
                + $"{(descriptor is null ? "missing" : descriptor.Shape)}.", role));
        }
    }

    private static int? BaseTimeEmbedDim(IReadOnlyDictionary<string, SafeTensorDescriptor> normalizedBase)
    {
        if (normalizedBase.TryGetValue("adaln_t_table", out SafeTensorDescriptor? table)
            && table.Shape.Rank == 2)
        {
            return checked((int)table.Shape[1]);
        }
        if (normalizedBase.TryGetValue("time_embedder.proj_out.weight", out SafeTensorDescriptor? projection)
            && projection.Shape.Rank == 2)
        {
            return checked((int)projection.Shape[0]);
        }
        return null;
    }

    private static VideoEffectiveSettings ResolveEffectiveSettings(VideoRequest request, VideoModelProfile profile,
        List<VideoPlanIssue> issues)
    {
        VideoDefaults defaults = profile.Defaults;
        ValidateLocked(request.Steps, defaults.Steps, VideoLockedFields.Steps, defaults.LockedFields,
            nameof(VideoRequest.Steps), issues);
        ValidateLocked(request.CfgScale, defaults.CfgScale, VideoLockedFields.CfgScale, defaults.LockedFields,
            nameof(VideoRequest.CfgScale), issues);
        ValidateLocked(request.FlowShift, defaults.FlowShift ?? 1f, VideoLockedFields.FlowShift, defaults.LockedFields,
            nameof(VideoRequest.FlowShift), issues);
        ValidateLocked(request.AudioFlowShift, defaults.AudioFlowShift ?? 1f, VideoLockedFields.AudioFlowShift,
            defaults.LockedFields, nameof(VideoRequest.AudioFlowShift), issues);
        ValidateLocked(request.Sampler, defaults.Sampler ?? "euler", VideoLockedFields.Sampler, defaults.LockedFields,
            nameof(VideoRequest.Sampler), issues);
        ValidateLocked(request.Scheduler, defaults.Scheduler ?? "normal", VideoLockedFields.Scheduler,
            defaults.LockedFields, nameof(VideoRequest.Scheduler), issues);

        if (profile.Acceleration == VideoAccelerationKind.Pdd && request.Steps is int pddSteps
            && pddSteps is not (4 or 6 or 8))
        {
            issues.Add(Error("video.pdd.nfe_unsupported", "PDD supports only 4, 6, or 8 transformer evaluations.",
                nameof(VideoRequest.Steps)));
        }
        int width = SafePositive(request.Width, defaults.Width,
            MiniMaxH3Geometry.MaxPixels / MiniMaxH3Geometry.CanvasMultiple);
        int height = SafePositive(request.Height, defaults.Height,
            MiniMaxH3Geometry.MaxPixels / MiniMaxH3Geometry.CanvasMultiple);
        int frames = SafePositive(request.Frames, defaults.Frames, int.MaxValue - 16);
        if ((defaults.LockedFields & VideoLockedFields.Geometry) != 0
            && ((request.Width is int requestWidth && requestWidth != defaults.Width)
                || (request.Height is int requestHeight && requestHeight != defaults.Height)))
        {
            issues.Add(Error("video.profile.geometry_locked",
                $"Profile '{profile.Id}' requires {defaults.Width}x{defaults.Height}.", "geometry"));
        }
        (width, height) = MiniMaxH3Geometry.ClampToMaxArea(width, height);
        frames = MiniMaxH3Geometry.AlignFrameCount(frames);
        ValidateResolvedGuideFrames(request, frames, issues);
        long seed = ResolveExecutionSeed(request.Seed);
        return new VideoEffectiveSettings
        {
            Width = width,
            Height = height,
            Frames = frames,
            Fps = SafePositive(request.Fps, defaults.Fps),
            Steps = SafePositive(request.Steps, defaults.Steps),
            CfgScale = SafeFinitePositive(request.CfgScale, defaults.CfgScale),
            FlowShift = SafeFinitePositive(request.FlowShift, defaults.FlowShift ?? 1f),
            AudioFlowShift = SafeFinitePositive(request.AudioFlowShift, defaults.AudioFlowShift ?? 1f),
            Sampler = string.Equals(request.Sampler, "euler", StringComparison.OrdinalIgnoreCase)
                ? request.Sampler! : defaults.Sampler ?? "euler",
            Scheduler = string.Equals(request.Scheduler, "normal", StringComparison.OrdinalIgnoreCase)
                ? request.Scheduler! : defaults.Scheduler ?? "normal",
            Seed = seed,
            ReferenceSizing = request.ReferenceSizing is VideoReferenceSizing sizing && Enum.IsDefined(sizing)
                ? sizing : defaults.ReferenceSizing,
            LockedFields = defaults.LockedFields,
        };
    }

    private static int SafePositive(int? requested, int? fallback, int maximum = int.MaxValue) =>
        requested is int value && value > 0 && value <= maximum ? value
            : fallback is int fallbackValue && fallbackValue > 0 && fallbackValue <= maximum ? fallbackValue : 1;

    private static float SafeFinitePositive(float? requested, float fallback) =>
        requested is float value && float.IsFinite(value) && value > 0f ? value : fallback;

    private static VideoEffectiveSettings EffectiveFromResolved(VideoRequest resolved, VideoLockedFields locked) =>
        new VideoEffectiveSettings
        {
            Width = resolved.Width!.Value,
            Height = resolved.Height!.Value,
            Frames = resolved.Frames!.Value,
            Fps = resolved.Fps!.Value,
            Steps = resolved.Steps!.Value,
            CfgScale = resolved.CfgScale!.Value,
            FlowShift = resolved.FlowShift,
            AudioFlowShift = resolved.AudioFlowShift,
            Sampler = resolved.Sampler,
            Scheduler = resolved.Scheduler,
            Seed = ResolveExecutionSeed(resolved.Seed),
            ReferenceSizing = resolved.ReferenceSizing ?? VideoReferenceSizing.Native,
            LockedFields = locked,
        };

    /// <summary>Normalizes the public 64-bit seed into the 31-bit space every existing video recipe actually sends
    /// to its diffusion pipeline. Planning, execution, and the audit summary must name the same seed.</summary>
    private static long ResolveExecutionSeed(long requested) =>
        RecipeRequestMapper.MapSeed(requested) ?? RandomNumberGenerator.GetInt32(int.MaxValue);

    private static void ValidateLocked(int? requested, int expected, VideoLockedFields field,
        VideoLockedFields locked, string name, List<VideoPlanIssue> issues)
    {
        if ((locked & field) != 0 && requested is int value && value != expected)
        {
            issues.Add(Error("video.profile.field_locked", $"{name} is locked to {expected} by the detected profile.", name));
        }
    }

    private static void ValidateLocked(float? requested, float expected, VideoLockedFields field,
        VideoLockedFields locked, string name, List<VideoPlanIssue> issues)
    {
        if ((locked & field) != 0 && requested is float value
            && (!float.IsFinite(value) || Math.Abs(value - expected) > 1e-6f))
        {
            issues.Add(Error("video.profile.field_locked",
                $"{name} is locked to {expected.ToString("R", CultureInfo.InvariantCulture)} by the detected profile.", name));
        }
    }

    private static void ValidateLocked(string? requested, string expected, VideoLockedFields field,
        VideoLockedFields locked, string name, List<VideoPlanIssue> issues)
    {
        if ((locked & field) != 0 && requested is not null
            && !string.Equals(requested, expected, StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.profile.field_locked", $"{name} is locked to '{expected}' by the detected profile.", name));
        }
    }

    private static HeaderSnapshot ReadHeader(string path)
    {
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, SafeTensorDescriptor> descriptors =
            new Dictionary<string, SafeTensorDescriptor>(loader.Descriptors, StringComparer.Ordinal);
        Dictionary<string, string> metadata = loader.Metadata is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
        return new HeaderSnapshot(descriptors, metadata);
    }

    private static void ValidateH3Structure(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        List<VideoPlanIssue> issues)
    {
        Dictionary<string, SafeTensorDescriptor> normalized = NormalizeDescriptors(descriptors);
        ValidateH3Shape(normalized, "video_patch_proj.weight", issues, 5376, 96);
        ValidateH3Shape(normalized, "video_patch_proj.bias", issues, 5376);
        ValidateH3Shape(normalized, "audio_patch_proj.weight", issues, 5376, 32);
        ValidateH3Shape(normalized, "audio_patch_proj.bias", issues, 5376);
        ValidateH3Shape(normalized, "condition_proj.weight", issues, 5376, 5120);
        ValidateH3Shape(normalized, "condition_proj.bias", issues, 5376);
        ValidateH3Shape(normalized, "rope.inv_freq", issues, 16);

        bool pruned = normalized.ContainsKey("adaln_t_table");
        int timeDim = pruned ? 8 : 2688;
        if (pruned)
        {
            ValidateH3Shape(normalized, "adaln_t_table", issues, 1025, 8);
            if (normalized.Keys.Any(key => key.StartsWith("time_embedder.", StringComparison.Ordinal)))
            {
                issues.Add(Error("video.checkpoint.time_embedding_ambiguous",
                    "A pruned H3 checkpoint may not mix adaln_t_table curves with a full time_embedder.",
                    nameof(ModelSpec.LocalPath)));
            }
        }
        else
        {
            ValidateH3Shape(normalized, "time_embedder.proj_in.weight", issues, 5376, 256);
            ValidateH3Shape(normalized, "time_embedder.proj_in.bias", issues, 5376);
            ValidateH3Shape(normalized, "time_embedder.proj_out.weight", issues, 2688, 5376);
            ValidateH3Shape(normalized, "time_embedder.proj_out.bias", issues, 2688);
        }

        for (int i = 0; i < 50; i++)
        {
            string block = $"blocks.{i}";
            ValidateH3Block(normalized, block, timeDim, issues);
        }
        ValidateIndexedFamily(normalized.Keys, "blocks.", 50, issues);

        for (int i = 0; i < 2; i++)
        {
            string block = $"token_refiner.blocks.{i}";
            ValidateH3Shape(normalized, block + ".norm1.weight", issues, 5376);
            ValidateH3Shape(normalized, block + ".norm2.weight", issues, 5376);
            ValidateH3Shape(normalized, block + ".attn.qkv_proj.weight", issues, 21504, 5376);
            ValidateH3Shape(normalized, block + ".attn.q_norm.weight", issues, 128);
            ValidateH3Shape(normalized, block + ".attn.k_norm.weight", issues, 128);
            ValidateH3Shape(normalized, block + ".attn.out_proj.weight", issues, 5376, 7168);
            ValidateH3Shape(normalized, block + ".mlp.fc1.weight", issues, 28672, 5376);
            ValidateH3Shape(normalized, block + ".mlp.fc2.weight", issues, 5376, 14336);
        }
        ValidateIndexedFamily(normalized.Keys, "token_refiner.blocks.", 2, issues);
        ValidateH3Shape(normalized, "token_refiner.final_norm.weight", issues, 5376);

        ValidateH3Shape(normalized, "final_layer.norm.weight", issues, 5376);
        ValidateH3Shape(normalized, "final_layer.adaln_proj.linear.weight", issues, 10752, timeDim);
        ValidateH3Shape(normalized, "final_layer.adaln_proj.linear.bias", issues, 10752);
        ValidateH3Shape(normalized, "final_layer.video_out.weight", issues, 96, 5376);
        ValidateH3Shape(normalized, "final_layer.video_out.bias", issues, 96);
        ValidateH3Shape(normalized, "final_layer.audio_out.weight", issues, 32, 5376);
        ValidateH3Shape(normalized, "final_layer.audio_out.bias", issues, 32);

        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in normalized)
        {
            if (!entry.Key.EndsWith(".weight", StringComparison.Ordinal))
            {
                continue;
            }
            string stem = entry.Key[..^".weight".Length];
            if (entry.Value.DType == DType.U8)
            {
                issues.Add(Error("video.checkpoint.quant_dtype_unsupported",
                    $"MiniMax-H3 transformer tensor '{entry.Key}' uses unsupported U8 packed weights.",
                    nameof(ModelSpec.LocalPath)));
            }
            if (entry.Value.DType == DType.I8)
            {
                ValidateInt8Companions(normalized, stem, entry.Value, "video.checkpoint", nameof(ModelSpec.LocalPath),
                    issues);
            }
            else if (entry.Value.DType == DType.F8E4M3 || entry.Value.DType == DType.F8E5M2)
            {
                ValidateFp8Companions(normalized, stem, issues);
            }
        }

        int gateCount = 0;
        for (int i = 0; i < 50; i++)
        {
            string key = $"blocks.{i}.attn.to_gate_compress.weight";
            if (normalized.TryGetValue(key, out SafeTensorDescriptor? gate))
            {
                gateCount++;
                if (gate.Shape.Rank != 2 || gate.Shape[0] != 7168 || gate.Shape[1] != 5376)
                {
                    issues.Add(Error("video.vsa.gate_shape_invalid", $"'{key}' must be [7168,5376], got {gate.Shape}.",
                        nameof(ModelSpec.LocalPath)));
                }
            }
        }
        if (gateCount is > 0 and < 50)
        {
            issues.Add(Error("video.vsa.gates_partial", $"VSA checkpoints require gates in all 50 blocks; found {gateCount}.",
                nameof(ModelSpec.LocalPath)));
        }
    }

    private static void ValidateH3Block(Dictionary<string, SafeTensorDescriptor> descriptors, string block,
        int timeDim, List<VideoPlanIssue> issues)
    {
        ValidateH3Shape(descriptors, block + ".norm1.weight", issues, 5376);
        ValidateH3Shape(descriptors, block + ".norm2.weight", issues, 5376);
        ValidateH3Shape(descriptors, block + ".attn.qkv_proj.weight", issues, 21504, 5376);
        ValidateH3Shape(descriptors, block + ".attn.q_norm.weight", issues, 128);
        ValidateH3Shape(descriptors, block + ".attn.k_norm.weight", issues, 128);
        ValidateH3Shape(descriptors, block + ".attn.out_proj.weight", issues, 5376, 7168);
        ValidateH3Shape(descriptors, block + ".mlp.fc1.weight", issues, 28672, 5376);
        ValidateH3Shape(descriptors, block + ".mlp.fc2.weight", issues, 5376, 14336);
        ValidateH3Shape(descriptors, block + ".adaln_proj.linear.weight", issues, 96768, timeDim);
        ValidateH3Shape(descriptors, block + ".adaln_proj.linear.bias", issues, 96768);
    }

    private static void ValidateIndexedFamily(IEnumerable<string> keys, string prefix, int expectedCount,
        List<VideoPlanIssue> issues)
    {
        HashSet<int> found = [];
        foreach (string key in keys)
        {
            if (!key.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            ReadOnlySpan<char> suffix = key.AsSpan(prefix.Length);
            int dot = suffix.IndexOf('.');
            if (dot > 0 && int.TryParse(suffix[..dot], NumberStyles.None, CultureInfo.InvariantCulture, out int index))
            {
                found.Add(index);
            }
        }
        int[] unexpected = found.Where(index => index < 0 || index >= expectedCount).Order().ToArray();
        if (unexpected.Length > 0)
        {
            issues.Add(Error("video.checkpoint.block_count_invalid",
                $"MiniMax-H3 family '{prefix}' must contain exactly indices 0..{expectedCount - 1}; unexpected: "
                + string.Join(',', unexpected) + ".", nameof(ModelSpec.LocalPath)));
        }
    }

    private static void ValidateVsaProfileStructure(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        VideoModelProfile profile, List<VideoPlanIssue> issues)
    {
        Dictionary<string, SafeTensorDescriptor> normalized = NormalizeDescriptors(descriptors);
        int gateCount = 0;
        for (int i = 0; i < 50; i++)
        {
            if (normalized.ContainsKey($"blocks.{i}.attn.to_gate_compress.weight"))
            {
                gateCount++;
            }
        }

        if (profile.Attention != VideoAttentionKind.Dense && gateCount != 50)
        {
            issues.Add(Error("video.vsa.gates_required",
                $"Sparse profile '{profile.Id}' requires valid gate projections in all 50 main blocks; found {gateCount}.",
                nameof(ModelSpec.LocalPath)));
        }
        else if (profile.Attention == VideoAttentionKind.Dense && gateCount == 50)
        {
            issues.Add(Error("video.vsa.profile_required",
                "The checkpoint contains all 50 VSA gates, but its hash or sidecar does not bind a supported VSA semantic profile.",
                nameof(ModelSpec.ProfileId)));
        }
    }

    /// <summary>Proves every ordinary H3 LoRA target against the selected checkpoint from headers alone. This is
    /// deliberately stricter than the generic merge path: a partial merge is a planning error, because otherwise an
    /// adapter trained for the full-width AdaLN build can begin an SSE request and then silently skip its incompatible
    /// curve-form targets.</summary>
    private static void ValidateOrdinaryLoraCompatibility(
        IReadOnlyDictionary<string, SafeTensorDescriptor> adapterDescriptors,
        IReadOnlyDictionary<string, SafeTensorDescriptor> baseDescriptors,
        string role, List<VideoPlanIssue> issues)
    {
        Dictionary<string, SafeTensorDescriptor> normalizedBase = NormalizeDescriptors(baseDescriptors);
        Dictionary<string, LoraHeaderPair> pairs = new(StringComparer.Ordinal);
        List<(string Code, string Message)> failures = [];
        bool officialDiffusers = adapterDescriptors.Keys.Any(key =>
            key.StartsWith("transformer_blocks.", StringComparison.Ordinal)
            || key.StartsWith("token_refiner.refiner_blocks.", StringComparison.Ordinal));

        foreach ((string key, SafeTensorDescriptor descriptor) in adapterDescriptors)
        {
            if (!TrySplitLoraRole(key, out string? root, out bool down))
            {
                continue;
            }
            if (!pairs.TryGetValue(root, out LoraHeaderPair? pair))
            {
                pair = new LoraHeaderPair();
                pairs[root] = pair;
            }
            if (down)
            {
                if (pair.Down is not null)
                {
                    failures.Add(("video.lora.target_duplicate",
                        $"LoRA target '{root}' carries more than one down/A tensor."));
                }
                pair.Down = descriptor;
            }
            else
            {
                if (pair.Up is not null)
                {
                    failures.Add(("video.lora.target_duplicate",
                        $"LoRA target '{root}' carries more than one up/B tensor."));
                }
                pair.Up = descriptor;
            }
        }

        if (pairs.Count == 0)
        {
            failures.Add(("video.lora.targets_missing",
                "The adapter header contains no recognized LoRA A/B or down/up target pairs."));
        }

        if (officialDiffusers)
        {
            HashSet<string> expected = ExpectedOfficialH3LoraRoots();
            string[] missing = expected.Except(pairs.Keys, StringComparer.Ordinal).Take(6).ToArray();
            string[] extra = pairs.Keys.Except(expected, StringComparer.Ordinal).Take(6).ToArray();
            if (missing.Length > 0 || extra.Length > 0 || pairs.Count != expected.Count)
            {
                failures.Add(("video.lora.diffusers_layout_incomplete",
                    $"Official MiniMax-H3 Diffusers layout must contain exactly {expected.Count} complete targets; "
                    + $"found {pairs.Count}. Missing: {string.Join(", ", missing)}; extra: {string.Join(", ", extra)}."));
            }
        }

        foreach ((string root, LoraHeaderPair pair) in pairs)
        {
            if (pair.Down is null || pair.Up is null)
            {
                failures.Add(("video.lora.target_pair_incomplete",
                    $"LoRA target '{root}' is missing its {(pair.Down is null ? "down/A" : "up/B")} tensor."));
                continue;
            }
            if (pair.Down.Shape.Rank != 2 || pair.Up.Shape.Rank != 2
                || pair.Down.Shape[0] <= 0 || pair.Down.Shape[0] != pair.Up.Shape[1])
            {
                failures.Add(("video.lora.rank_mismatch",
                    $"LoRA target '{root}' must use compatible rank-two A/B matrices; got "
                    + $"{pair.Down.Shape} and {pair.Up.Shape}."));
                continue;
            }
            if (pair.Down.DType != pair.Up.DType || pair.Down.DType != DType.F32
                && pair.Down.DType != DType.F16 && pair.Down.DType != DType.BF16)
            {
                failures.Add(("video.lora.dtype_incompatible",
                    $"LoRA target '{root}' must use matching F32/F16/BF16 matrices; got "
                    + $"{pair.Down.DType}/{pair.Up.DType}."));
                continue;
            }
            if (!TryResolveH3LoraTarget(root, normalizedBase, out string? target,
                out long expectedRows, out long expectedColumns))
            {
                failures.Add(("video.lora.target_missing",
                    $"LoRA target '{root}' does not map to a weight in the selected MiniMax-H3 checkpoint."));
                continue;
            }
            if (pair.Up.Shape[0] != expectedRows || pair.Down.Shape[1] != expectedColumns)
            {
                failures.Add(("video.lora.target_shape_mismatch",
                    $"LoRA target '{root}' produces [{pair.Up.Shape[0]},{pair.Down.Shape[1]}], but selected weight "
                    + $"'{target}' requires [{expectedRows},{expectedColumns}]."));
            }
        }

        const int maxDetailedFailures = 8;
        foreach ((string code, string message) in failures.Take(maxDetailedFailures))
        {
            issues.Add(Error(code, message, role));
        }
        if (failures.Count > maxDetailedFailures)
        {
            issues.Add(Error("video.lora.incompatible_summary",
                $"The adapter has {failures.Count} incompatible targets; only the first {maxDetailedFailures} are shown.",
                role));
        }
    }

    private static bool TrySplitLoraRole(string key, out string root, out bool down)
    {
        (string Suffix, bool Down)[] suffixes =
        [
            (".lora_A.default.weight", true), (".lora_B.default.weight", false),
            (".lora_down.weight", true), (".lora_up.weight", false),
            (".lora_A.weight", true), (".lora_B.weight", false),
            (".lora_down", true), (".lora_up", false),
        ];
        foreach ((string suffix, bool isDown) in suffixes)
        {
            if (!key.EndsWith(suffix, StringComparison.Ordinal))
            {
                continue;
            }
            root = key[..^suffix.Length];
            down = isDown;
            return true;
        }
        root = string.Empty;
        down = false;
        return false;
    }

    private static bool TryResolveH3LoraTarget(string rawRoot,
        IReadOnlyDictionary<string, SafeTensorDescriptor> normalizedBase, out string target,
        out long expectedRows, out long expectedColumns)
    {
        string root = rawRoot;
        foreach (string prefix in new[] { "model.diffusion_model.", "diffusion_model.", "transformer." })
        {
            if (root.StartsWith(prefix, StringComparison.Ordinal))
            {
                root = root[prefix.Length..];
                break;
            }
        }

        bool official = root.StartsWith("transformer_blocks.", StringComparison.Ordinal)
            || root.StartsWith("token_refiner.refiner_blocks.", StringComparison.Ordinal);
        if (official)
        {
            root = root.Replace("transformer_blocks.", "blocks.", StringComparison.Ordinal)
                .Replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", StringComparison.Ordinal);
            string[] splitParts = ["q", "k", "v"];
            foreach (string part in splitParts)
            {
                string suffix = $".attn.to_{part}";
                if (!root.EndsWith(suffix, StringComparison.Ordinal))
                {
                    continue;
                }
                target = root[..^suffix.Length] + ".attn.qkv_proj.weight";
                if (normalizedBase.TryGetValue(target, out SafeTensorDescriptor? fused)
                    && fused.Shape.Rank == 2 && fused.Shape[0] % 3 == 0)
                {
                    expectedRows = fused.Shape[0] / 3;
                    expectedColumns = fused.Shape[1];
                    return true;
                }
                expectedRows = expectedColumns = 0;
                return false;
            }
            (string Source, string Target)[] mappings =
            [
                (".attn.to_out.0", ".attn.out_proj"),
                (".ff.net.0.proj", ".mlp.fc1"),
                (".ff.net.2", ".mlp.fc2"),
            ];
            foreach ((string source, string replacement) in mappings)
            {
                if (root.EndsWith(source, StringComparison.Ordinal))
                {
                    root = root[..^source.Length] + replacement;
                    break;
                }
            }
        }
        root = root.Replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", StringComparison.Ordinal);
        target = root + ".weight";
        if (normalizedBase.TryGetValue(target, out SafeTensorDescriptor? descriptor)
            && descriptor.Shape.Rank == 2)
        {
            expectedRows = descriptor.Shape[0];
            expectedColumns = descriptor.Shape[1];
            return true;
        }
        expectedRows = expectedColumns = 0;
        return false;
    }

    private static HashSet<string> ExpectedOfficialH3LoraRoots()
    {
        HashSet<string> expected = new(StringComparer.Ordinal);
        static void AddBlock(HashSet<string> set, string root)
        {
            set.Add(root + ".attn.to_q");
            set.Add(root + ".attn.to_k");
            set.Add(root + ".attn.to_v");
            set.Add(root + ".attn.to_out.0");
            set.Add(root + ".ff.net.0.proj");
            set.Add(root + ".ff.net.2");
        }
        for (int index = 0; index < 50; index++)
        {
            AddBlock(expected, $"transformer_blocks.{index}");
        }
        for (int index = 0; index < 2; index++)
        {
            AddBlock(expected, $"token_refiner.refiner_blocks.{index}");
        }
        return expected;
    }

    private sealed class LoraHeaderPair
    {
        public SafeTensorDescriptor? Down { get; set; }
        public SafeTensorDescriptor? Up { get; set; }
    }

    private static void ValidatePddHeader(HeaderSnapshot header, VideoKnownArtifact artifact,
        List<VideoPlanIssue> issues)
    {
        ValidatePddMetadataInt(header.Metadata, "pdd_num_steps", MiniMaxH3PddSchedule.PublishedFineSteps, issues);
        ValidatePddMetadataInt(header.Metadata, "pdd_block_size", MiniMaxH3PddSchedule.PublishedBlockSize, issues);
        ValidatePddPositiveMetadata(header.Metadata, "lora_rank", issues);
        ValidatePddPositiveMetadata(header.Metadata, "lora_alpha", issues);

        string? taskText = null;
        foreach (string key in new[] { "pdd_task", "pdd_partition", "hartsy.pdd.task" })
        {
            if (header.Metadata.TryGetValue(key, out taskText))
            {
                break;
            }
        }
        if (taskText is null)
        {
            issues.Add(Error("video.pdd.task_missing",
                "PDD metadata must preserve a pdd_task, pdd_partition, or hartsy.pdd.task binding.",
                nameof(VideoRequest.Loras)));
        }
        else
        {
            VideoTaskFamily metadataTask = NormalizePddTask(taskText);
            if (metadataTask is not (VideoTaskFamily.Fl2Va or VideoTaskFamily.Ref2Va))
            {
                issues.Add(Error("video.pdd.task_invalid",
                    $"PDD metadata task '{taskText}' is not fl2va or ref2va.", nameof(VideoRequest.Loras)));
            }
            else if (metadataTask != artifact.Task)
            {
                issues.Add(Error("video.pdd.task_hash_conflict",
                    $"PDD metadata binds {metadataTask}, but artifact hash/profile binds {artifact.Task}.",
                    nameof(VideoRequest.Loras)));
            }
        }

        SafeTensorDescriptor? videoWeight = ResolvePddDescriptor(header.Descriptors,
            ["proj_out.weight", "final_layer.video_out.weight", "final_layer.video_out.diff",
                "diffusion_model.final_layer.video_out.diff"], "video weight", issues);
        SafeTensorDescriptor? videoBias = ResolvePddDescriptor(header.Descriptors,
            ["proj_out.bias", "final_layer.video_out.bias", "final_layer.video_out.diff_b",
                "diffusion_model.final_layer.video_out.diff_b"], "video bias", issues);
        SafeTensorDescriptor? audioWeight = ResolvePddDescriptor(header.Descriptors,
            ["audio_proj_out.weight", "final_layer.audio_out.weight", "final_layer.audio_out.diff",
                "diffusion_model.final_layer.audio_out.diff"], "audio weight", issues);
        SafeTensorDescriptor? audioBias = ResolvePddDescriptor(header.Descriptors,
            ["audio_proj_out.bias", "final_layer.audio_out.bias", "final_layer.audio_out.diff_b",
                "diffusion_model.final_layer.audio_out.diff_b"], "audio bias", issues);
        if (videoWeight is null || videoBias is null || audioWeight is null || audioBias is null)
        {
            return;
        }
        if (new[] { videoWeight, videoBias, audioWeight, audioBias }.Any(descriptor => descriptor.DType != DType.F32))
        {
            issues.Add(Error("video.pdd.bank_dtype_invalid",
                "PDD video/audio projection banks must be F32 so runtime fusion never performs an implicit multi-gigabyte cast.",
                nameof(VideoRequest.Loras)));
        }
        bool rankThree = videoWeight.Shape.Rank == 3;
        bool flattened = videoWeight.Shape.Rank == 2;
        if (!rankThree && !flattened)
        {
            issues.Add(Error("video.pdd.bank_rank_invalid",
                $"PDD video head bank must be rank three or known flattened rank two; got {videoWeight.Shape}.",
                nameof(VideoRequest.Loras)));
            return;
        }
        if (rankThree)
        {
            ValidateExactPddShape(videoWeight, [32, 96, 5376], "video weight", issues);
            ValidateExactPddShape(videoBias, [32, 96], "video bias", issues);
            ValidateExactPddShape(audioWeight, [32, 32, 5376], "audio weight", issues);
            ValidateExactPddShape(audioBias, [32, 32], "audio bias", issues);
            return;
        }

        string? layout = header.Metadata.TryGetValue("hartsy.pdd.head_layout", out string? hartsyLayout)
            ? hartsyLayout
            : header.Metadata.TryGetValue("pdd_head_layout", out string? pddLayout) ? pddLayout : null;
        if (!string.Equals(layout, "base_plus_offsets_flat", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(layout, "flattened_offsets", StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.pdd.flattened_ambiguous",
                "Flattened PDD banks are ambiguous and require hash-bound base-plus-offset head-layout metadata.",
                nameof(VideoRequest.Loras)));
        }
        ValidateExactPddShape(videoWeight, [3072, 5376], "flattened video weight", issues);
        ValidateExactPddShape(videoBias, [3072], "flattened video bias", issues);
        ValidateExactPddShape(audioWeight, [1024, 5376], "flattened audio weight", issues);
        ValidateExactPddShape(audioBias, [1024], "flattened audio bias", issues);
    }

    private static SafeTensorDescriptor? ResolvePddDescriptor(
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, string[] aliases, string role,
        List<VideoPlanIssue> issues)
    {
        string[] found = aliases.Where(descriptors.ContainsKey).ToArray();
        if (found.Length != 1)
        {
            issues.Add(Error("video.pdd.bank_alias_ambiguous",
                $"PDD adapter must expose exactly one {role}; found {found.Length}: {string.Join(", ", found)}.",
                nameof(VideoRequest.Loras)));
            return null;
        }
        return descriptors[found[0]];
    }

    private static void ValidateExactPddShape(SafeTensorDescriptor descriptor, long[] expected, string role,
        List<VideoPlanIssue> issues)
    {
        if (descriptor.Shape.Rank == expected.Length
            && expected.Select((dimension, index) => descriptor.Shape[index] == dimension).All(matches => matches))
        {
            return;
        }
        issues.Add(Error("video.pdd.bank_shape_invalid",
            $"PDD {role} must be [{string.Join(',', expected)}], got {descriptor.Shape}.",
            nameof(VideoRequest.Loras)));
    }

    private static void ValidatePddMetadataInt(IReadOnlyDictionary<string, string> metadata, string key,
        int expected, List<VideoPlanIssue> issues)
    {
        if (!metadata.TryGetValue(key, out string? text))
        {
            issues.Add(Error("video.pdd.metadata_missing",
                $"PDD metadata must preserve {key}={expected}.", nameof(VideoRequest.Loras)));
            return;
        }
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) || value != expected)
        {
            issues.Add(Error("video.pdd.metadata_invalid",
                $"PDD metadata {key} must equal {expected}; got '{text}'.", nameof(VideoRequest.Loras)));
        }
    }

    private static void ValidatePddPositiveMetadata(IReadOnlyDictionary<string, string> metadata, string key,
        List<VideoPlanIssue> issues)
    {
        if (!metadata.TryGetValue(key, out string? text))
        {
            issues.Add(Error("video.pdd.metadata_missing",
                $"PDD metadata must preserve a finite positive {key} value.", nameof(VideoRequest.Loras)));
            return;
        }
        if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
            || !double.IsFinite(value) || value <= 0.0)
        {
            issues.Add(Error("video.pdd.metadata_invalid",
                $"PDD metadata {key} must be finite and positive; got '{text}'.", nameof(VideoRequest.Loras)));
        }
    }

    private static Dictionary<string, SafeTensorDescriptor> NormalizeDescriptors(
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        Dictionary<string, SafeTensorDescriptor> normalized = new Dictionary<string, SafeTensorDescriptor>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in descriptors)
        {
            string key = entry.Key;
            foreach (string prefix in new[] { "model.diffusion_model.", "diffusion_model.", "transformer." })
            {
                if (key.StartsWith(prefix, StringComparison.Ordinal))
                {
                    key = key[prefix.Length..];
                    break;
                }
            }
            normalized[key] = entry.Value;
        }
        return normalized;
    }

    private static void ValidateH3Shape(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, string key,
        List<VideoPlanIssue> issues, params long[] expected)
    {
        if (!descriptors.TryGetValue(key, out SafeTensorDescriptor? descriptor))
        {
            issues.Add(Error("video.checkpoint.tensor_missing", $"MiniMax-H3 checkpoint is missing '{key}'.",
                nameof(ModelSpec.LocalPath)));
            return;
        }
        if (!ShapeEquals(descriptor, expected))
        {
            issues.Add(Error("video.checkpoint.tensor_shape_invalid",
                $"MiniMax-H3 tensor '{key}' must be [{string.Join(',', expected)}], got {descriptor.Shape}.",
                nameof(ModelSpec.LocalPath)));
        }
    }

    private static bool ShapeEquals(SafeTensorDescriptor descriptor, params long[] expected) =>
        descriptor.Shape.Rank == expected.Length
        && expected.Select((dimension, index) => descriptor.Shape[index] == dimension).All(matches => matches);

    private static long ElementCount(SafeTensorDescriptor descriptor)
    {
        long count = 1;
        for (int i = 0; i < descriptor.Shape.Rank; i++)
        {
            count = checked(count * descriptor.Shape[i]);
        }
        return count;
    }

    private static void ValidateInt8Companions(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string stem, SafeTensorDescriptor weight, string codePrefix, string field, List<VideoPlanIssue> issues)
    {
        if (!descriptors.TryGetValue(stem + ".weight_scale", out SafeTensorDescriptor? scale)
            || !descriptors.TryGetValue(stem + ".comfy_quant", out SafeTensorDescriptor? quant))
        {
            issues.Add(Error(codePrefix + ".quant_companion_missing",
                $"Quantized tensor '{stem}.weight' is missing weight_scale or comfy_quant.", field));
            return;
        }
        long rows = weight.Shape.Rank > 0 ? weight.Shape[0] : 0;
        long scaleElements;
        try
        {
            scaleElements = ElementCount(scale);
        }
        catch (OverflowException)
        {
            scaleElements = -1;
        }
        if (scale.DType != DType.F32 || scaleElements is not 1 && scaleElements != rows)
        {
            issues.Add(Error(codePrefix + ".quant_scale_invalid",
                $"'{stem}.weight_scale' must be F32 with one value or one per output row ({rows}); got "
                + $"{scale.DType} {scale.Shape}.", field));
        }
        long quantElements;
        try
        {
            quantElements = ElementCount(quant);
        }
        catch (OverflowException)
        {
            quantElements = -1;
        }
        if (quant.DType != DType.U8 || quant.Shape.Rank != 1 || quantElements is < 2 or > 4096)
        {
            issues.Add(Error(codePrefix + ".quant_descriptor_invalid",
                $"'{stem}.comfy_quant' must be a 2..4096-byte U8 descriptor; got {quant.DType} {quant.Shape}.",
                field));
        }
    }

    private static void ValidateFp8Companions(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string stem, List<VideoPlanIssue> issues)
    {
        if (!descriptors.TryGetValue(stem + ".comfy_quant", out SafeTensorDescriptor? quant)
            || quant.DType != DType.U8 || quant.Shape.Rank != 1)
        {
            issues.Add(Error("video.checkpoint.quant_descriptor_invalid",
                $"FP8 tensor '{stem}.weight' requires a one-dimensional U8 comfy_quant descriptor.",
                nameof(ModelSpec.LocalPath)));
        }
        if (descriptors.TryGetValue(stem + ".weight_scale", out SafeTensorDescriptor? scale)
            && (scale.DType != DType.F32 || ElementCount(scale) != 1))
        {
            issues.Add(Error("video.checkpoint.quant_scale_invalid",
                $"FP8 tensor '{stem}.weight_scale' must be an F32 scalar; got {scale.DType} {scale.Shape}.",
                nameof(ModelSpec.LocalPath)));
        }
    }

    /// <summary>Classifies a component from dtype and quantization-companion structure without reading tensor data.</summary>
    internal static string DetectFormat(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        bool hasNvfp4 = descriptors.Values.Any(descriptor => descriptor.DType == DType.U8)
            && descriptors.Keys.Any(key => key.EndsWith(".weight_scale_2", StringComparison.Ordinal));
        if (hasNvfp4)
        {
            bool hasAwq = descriptors.Keys.Any(key => key.EndsWith(".pre_quant_scale", StringComparison.Ordinal));
            return hasAwq ? "nvfp4-awq" : "nvfp4";
        }
        bool hasInt8 = descriptors.Values.Any(descriptor => descriptor.DType == DType.I8);
        bool hasConvRot = descriptors.Keys.Any(key => key.EndsWith(".comfy_quant", StringComparison.Ordinal));
        if (hasInt8)
        {
            return hasConvRot ? "int8-convrot" : "int8";
        }
        if (descriptors.Values.Any(descriptor => descriptor.DType == DType.F8E4M3 || descriptor.DType == DType.F8E5M2))
        {
            return "fp8";
        }
        if (descriptors.Values.Any(descriptor => descriptor.DType == DType.BF16))
        {
            return "bf16";
        }
        if (descriptors.Values.Any(descriptor => descriptor.DType == DType.F16))
        {
            return "fp16";
        }
        return "fp32";
    }

    private static async Task<VideoProfileSidecar?> ReadSidecarAsync(string checkpointPath, string hash,
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, List<VideoPlanIssue> issues,
        CancellationToken cancel)
    {
        string appended = checkpointPath + ".hartsy-video-profile.json";
        string changed = Path.ChangeExtension(checkpointPath, ".hartsy-video-profile.json");
        string? sidecarPath = File.Exists(appended) ? appended : File.Exists(changed) ? changed : null;
        if (sidecarPath is null)
        {
            return null;
        }
        try
        {
            await using FileStream stream = File.OpenRead(sidecarPath);
            VideoProfileSidecar? sidecar = await JsonSerializer.DeserializeAsync(stream,
                VideoPlanningJsonContext.Default.VideoProfileSidecar, cancel).ConfigureAwait(false);
            if (sidecar is null)
            {
                issues.Add(Error("video.profile.sidecar_empty", $"Profile sidecar '{sidecarPath}' is empty.",
                    nameof(ModelSpec.ProfileId)));
                return null;
            }
            if (!string.Equals(sidecar.Sha256, hash, StringComparison.OrdinalIgnoreCase))
            {
                issues.Add(Error("video.profile.sidecar_hash_mismatch",
                    $"Profile sidecar binds SHA-256 {sidecar.Sha256}, but the checkpoint hashes to {hash}.",
                    nameof(ModelSpec.ProfileId)));
                return null;
            }
            if (!ValidateSidecar(sidecar, descriptors, issues))
            {
                return null;
            }
            return sidecar;
        }
        catch (Exception ex) when (ex is IOException or JsonException or NotSupportedException)
        {
            issues.Add(Error("video.profile.sidecar_invalid", $"Could not parse '{sidecarPath}': {ex.Message}",
                nameof(ModelSpec.ProfileId)));
            return null;
        }
    }

    private static bool ValidateSidecar(VideoProfileSidecar sidecar,
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, List<VideoPlanIssue> issues)
    {
        int issueCount = issues.Count;
        if (string.IsNullOrWhiteSpace(sidecar.ProfileId) || sidecar.Steps <= 0)
        {
            issues.Add(Error("video.profile.sidecar_invalid",
                "Profile sidecar needs a non-empty profileId and a positive steps value.", nameof(ModelSpec.ProfileId)));
        }
        if (!Enum.IsDefined(sidecar.Task) || sidecar.Task == VideoTaskFamily.Unknown)
        {
            issues.Add(Error("video.profile.sidecar_task_invalid",
                "A sidecar must bind exactly one published H3 task: T2VA, FL2VA, Ref2VA, or Hybrid.",
                nameof(ModelSpec.ProfileId)));
        }
        if (!Enum.IsDefined(sidecar.Acceleration) || sidecar.Acceleration == VideoAccelerationKind.Pdd)
        {
            issues.Add(Error("video.profile.sidecar_acceleration_invalid",
                "A main-checkpoint sidecar may certify None, Turbo, or VSA; PDD is selected only by a validated adapter.",
                nameof(ModelSpec.ProfileId)));
        }
        if (!Enum.IsDefined(sidecar.Attention) || !Enum.IsDefined(sidecar.ReferenceSizing))
        {
            issues.Add(Error("video.profile.sidecar_enum_invalid",
                "The sidecar contains an unknown attention or reference-sizing value.", nameof(ModelSpec.ProfileId)));
        }
        if (!float.IsFinite(sidecar.CfgScale) || Math.Abs(sidecar.CfgScale - 1f) > 1e-6f
            || !float.IsFinite(sidecar.FlowShift) || sidecar.FlowShift <= 0f
            || !float.IsFinite(sidecar.AudioFlowShift) || sidecar.AudioFlowShift <= 0f)
        {
            issues.Add(Error("video.profile.sidecar_numeric_invalid",
                "H3 sidecars require CFG 1 and finite positive video/audio flow shifts.", nameof(ModelSpec.ProfileId)));
        }
        if (!string.Equals(sidecar.Sampler, "euler", StringComparison.OrdinalIgnoreCase)
            || !string.Equals(sidecar.Scheduler, "normal", StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.profile.sidecar_sampler_invalid",
                "H3 sidecars may certify only the Euler sampler with the normal scheduler.",
                nameof(ModelSpec.ProfileId)));
        }

        bool hasWidth = sidecar.Width is not null;
        bool hasHeight = sidecar.Height is not null;
        if (hasWidth != hasHeight || hasWidth && (sidecar.Width!.Value <= 0 || sidecar.Height!.Value <= 0
            || sidecar.Width.Value % MiniMaxH3Geometry.CanvasMultiple != 0
            || sidecar.Height.Value % MiniMaxH3Geometry.CanvasMultiple != 0
            || (long)sidecar.Width.Value * sidecar.Height.Value > MiniMaxH3Geometry.MaxPixels))
        {
            issues.Add(Error("video.profile.sidecar_geometry_invalid",
                $"Sidecar geometry must provide both axes, use {MiniMaxH3Geometry.CanvasMultiple}-pixel multiples, "
                + $"and not exceed {MiniMaxH3Geometry.MaxPixels} pixels.", nameof(ModelSpec.ProfileId)));
        }

        bool sparseAttention = sidecar.Attention != VideoAttentionKind.Dense;
        if (sidecar.Acceleration == VideoAccelerationKind.Vsa)
        {
            if (!sparseAttention)
            {
                issues.Add(Error("video.profile.sidecar_attention_invalid",
                    "VSA acceleration requires a published sparse-attention semantic profile.",
                    nameof(ModelSpec.ProfileId)));
            }
            if (!sparseAttention || sidecar.Task != VideoTaskFamily.T2Va || sidecar.Steps != 4
                || Math.Abs(sidecar.FlowShift - 12f) > 1e-6f
                || Math.Abs(sidecar.AudioFlowShift - 3f) > 1e-6f)
            {
                issues.Add(Error("video.profile.sidecar_vsa_matrix_invalid",
                    "VSA sidecars must bind T2VA, a published sparse-attention semantic profile, four evaluations, "
                    + "CFG 1, and exact 12/3 video/audio shifts.", nameof(ModelSpec.ProfileId)));
            }
        }
        else if (sparseAttention)
        {
            issues.Add(Error("video.profile.sidecar_attention_invalid",
                "Sparse attention can be activated only by a VSA acceleration profile.", nameof(ModelSpec.ProfileId)));
        }
        if (sidecar.Task == VideoTaskFamily.Hybrid && sparseAttention)
        {
            issues.Add(Error("video.profile.sidecar_hybrid_invalid",
                "Hybrid conditioning is certified only with dense attention.", nameof(ModelSpec.ProfileId)));
        }

        Dictionary<string, SafeTensorDescriptor> normalized = NormalizeDescriptors(descriptors);
        bool hasAllGates = Enumerable.Range(0, 50)
            .All(index => normalized.ContainsKey($"blocks.{index}.attn.to_gate_compress.weight"));
        if (sidecar.Acceleration == VideoAccelerationKind.Vsa && !hasAllGates)
        {
            issues.Add(Error("video.profile.sidecar_vsa_gates_missing",
                "A VSA sidecar cannot certify a checkpoint without all 50 learned gate projections.",
                nameof(ModelSpec.ProfileId)));
        }
        return issues.Skip(issueCount).All(issue => issue.Severity != VideoPlanIssueSeverity.Error);
    }

    private static void AddFilenameHintIssues(string path, VideoModelProfile profile, bool hasSidecar,
        List<VideoPlanIssue> issues)
    {
        string name = Path.GetFileName(path);
        string lower = name.ToLowerInvariant();
        string[] hints = ["turbo", "pdd", "vsa", "hybrid"];
        foreach (string hint in hints)
        {
            if (!lower.Contains(hint, StringComparison.Ordinal))
            {
                continue;
            }
            issues.Add(Warning("video.profile.filename_hint_ignored",
                $"Filename suggests '{hint}', but filenames never activate an acceleration or Hybrid profile."));
            if (!hasSidecar && profile.Acceleration == VideoAccelerationKind.None)
            {
                issues.Add(Error("video.profile.sidecar_required",
                    $"Unknown checkpoint '{name}' appears to claim '{hint}'; add a hash-bound .hartsy-video-profile.json sidecar.",
                    nameof(ModelSpec.ProfileId)));
            }
        }
    }

    private static async Task AddComponentAsync(string role, string path, bool hashForManifest,
        Dictionary<string, string> componentPaths, Dictionary<string, string> componentFormats,
        Dictionary<string, string> artifactHashes, List<VideoPlanIssue> issues, CancellationToken cancel)
    {
        if (!File.Exists(path))
        {
            issues.Add(Error("video.component.missing", $"Video component '{role}' was not found at '{path}'.", role));
            return;
        }
        componentPaths[role] = path;
        HeaderSnapshot? snapshot = TryReadComponentHeader(path, issues, role);
        componentFormats[role] = snapshot is null ? "invalid" : DetectFormat(snapshot.Descriptors);
        if (snapshot is not null)
        {
            ValidateComponentStructure(role, snapshot.Descriptors, issues);
        }
        if (!hashForManifest)
        {
            return;
        }
        try
        {
            string hash = await VideoCheckpointHashCache.GetSha256Async(path, cancel).ConfigureAwait(false);
            artifactHashes[role] = hash;
            // Structural VAE validation is necessary but not sufficient: a known hash registered for another role
            // is a high-signal wiring mistake and must fail before its tensors enter the recipe cache.
            if (VideoProfileManifest.TryGetByHash(hash, out VideoKnownArtifact? artifact) && artifact is not null
                && artifact.Role != VideoProfileArtifactRole.VideoVae)
            {
                issues.Add(Error("video.component.wrong_artifact_role",
                    $"'{artifact.DisplayName}' cannot be used as the video VAE.", role));
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            issues.Add(Error("video.component.hash_failed", $"Could not hash component '{path}': {ex.Message}", role));
        }
    }

    private static HeaderSnapshot? TryReadComponentHeader(string path, List<VideoPlanIssue> issues, string role)
    {
        try
        {
            return ReadHeader(path);
        }
        catch (Exception ex) when (ex is IOException or JsonException or InvalidOperationException
            or HartsyInference.Core.Exceptions.HartsyInferenceException)
        {
            issues.Add(Error("video.component.header_invalid", $"Could not inspect component '{path}': {ex.Message}", role));
            return null;
        }
    }

    private static void ValidateComponentStructure(string role,
        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, List<VideoPlanIssue> issues)
    {
        switch (role)
        {
            case "videoVae":
                ValidateVideoVaeStructure(descriptors, issues, role);
                break;
            case "audioVae":
                ValidateAudioVaeStructure(descriptors, issues, role);
                break;
            case "textEncoder":
                ValidateTextEncoderStructure(descriptors, issues, role);
                break;
        }
    }

    private static void ValidateVideoVaeStructure(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        List<VideoPlanIssue> issues, string role)
    {
        ValidateComponentShape(descriptors, "latents_mean", role, issues, 24);
        ValidateComponentShape(descriptors, "latents_std", role, issues, 24);
        ValidateComponentShape(descriptors, "post_quant_conv.weight", role, issues, 24, 24, 1, 1, 1);
        ValidateComponentShape(descriptors, "post_quant_conv.bias", role, issues, 24);
        ValidateComponentShape(descriptors, "decoder.x_embedder.weight", role, issues, 2048, 24);
        ValidateComponentShape(descriptors, "decoder.x_embedder.bias", role, issues, 2048);
        ValidateComponentShape(descriptors, "decoder.register_tokens", role, issues, 1, 4, 2048);
        ValidateComponentShape(descriptors, "decoder.norm_out.weight", role, issues, 2048);
        ValidateComponentShape(descriptors, "decoder.norm_out.bias", role, issues, 2048);
        ValidateComponentShape(descriptors, "decoder.proj_out.weight", role, issues, 3072, 2048);
        ValidateComponentShape(descriptors, "decoder.proj_out.bias", role, issues, 3072);
        for (int i = 0; i < 36; i++)
        {
            string block = $"decoder.transformer_blocks.{i}";
            ValidateComponentShape(descriptors, block + ".norm1.weight", role, issues, 2048);
            ValidateComponentShape(descriptors, block + ".norm2.weight", role, issues, 2048);
            ValidateComponentShape(descriptors, block + ".scale1", role, issues, 2048);
            ValidateComponentShape(descriptors, block + ".scale2", role, issues, 2048);
            ValidateComponentShape(descriptors, block + ".attn.to_qkv.weight", role, issues, 6144, 2048);
            ValidateComponentShape(descriptors, block + ".attn.to_qkv.bias", role, issues, 6144);
            ValidateComponentShape(descriptors, block + ".attn.to_out.weight", role, issues, 2048, 2048);
            ValidateComponentShape(descriptors, block + ".attn.to_out.bias", role, issues, 2048);
            ValidateComponentShape(descriptors, block + ".ff.w1.weight", role, issues, 16384, 2048);
            ValidateComponentShape(descriptors, block + ".ff.w1.bias", role, issues, 16384);
            ValidateComponentShape(descriptors, block + ".ff.w2.weight", role, issues, 2048, 8192);
            ValidateComponentShape(descriptors, block + ".ff.w2.bias", role, issues, 2048);
        }
        ValidateComponentIndexedFamily(descriptors.Keys, "decoder.transformer_blocks.", 36, role, issues);

        ValidateComponentShape(descriptors, "encoder.conv_in.weight", role, issues, 128, 3, 3, 3, 3);
        ValidateComponentShape(descriptors, "encoder.conv_in.bias", role, issues, 128);
        int[] channels = [128, 256, 256, 512, 512, 1024];
        int inputChannels = 128;
        for (int stage = 0; stage < channels.Length; stage++)
        {
            int outputChannels = channels[stage];
            for (int blockIndex = 0; blockIndex < 2; blockIndex++)
            {
                string block = $"encoder.down.{stage}.block.{blockIndex}";
                int blockInput = blockIndex == 0 ? inputChannels : outputChannels;
                ValidateComponentShape(descriptors, block + ".norm1.weight", role, issues, blockInput);
                ValidateComponentShape(descriptors, block + ".norm1.bias", role, issues, blockInput);
                ValidateComponentShape(descriptors, block + ".conv1.weight", role, issues,
                    outputChannels, blockInput, 3, 3, 3);
                ValidateComponentShape(descriptors, block + ".conv1.bias", role, issues, outputChannels);
                ValidateComponentShape(descriptors, block + ".norm2.weight", role, issues, outputChannels);
                ValidateComponentShape(descriptors, block + ".norm2.bias", role, issues, outputChannels);
                ValidateComponentShape(descriptors, block + ".conv2.weight", role, issues,
                    outputChannels, outputChannels, 3, 3, 3);
                ValidateComponentShape(descriptors, block + ".conv2.bias", role, issues, outputChannels);
                if (blockInput != outputChannels)
                {
                    ValidateComponentShape(descriptors, block + ".nin_shortcut.weight", role, issues,
                        outputChannels, blockInput, 1, 1, 1);
                    ValidateComponentShape(descriptors, block + ".nin_shortcut.bias", role, issues, outputChannels);
                }
            }
            if (stage < 4)
            {
                string downsample = $"encoder.down.{stage}.downsample.conv";
                ValidateComponentShape(descriptors, downsample + ".weight", role, issues,
                    outputChannels, outputChannels, 3, 3, 3);
                ValidateComponentShape(descriptors, downsample + ".bias", role, issues, outputChannels);
            }
            inputChannels = outputChannels;
        }
        ValidateComponentIndexedFamily(descriptors.Keys, "encoder.down.", 6, role, issues);
        ValidateComponentShape(descriptors, "encoder.norm_out.weight", role, issues, 1024);
        ValidateComponentShape(descriptors, "encoder.norm_out.bias", role, issues, 1024);
        ValidateComponentShape(descriptors, "encoder.conv_out.weight", role, issues, 48, 1024, 3, 3, 3);
        ValidateComponentShape(descriptors, "encoder.conv_out.bias", role, issues, 48);
        ValidateComponentShape(descriptors, "quant_conv.weight", role, issues, 48, 48, 1, 1, 1);
        ValidateComponentShape(descriptors, "quant_conv.bias", role, issues, 48);

        foreach ((string key, SafeTensorDescriptor descriptor) in descriptors)
        {
            if (!key.EndsWith(".weight", StringComparison.Ordinal) || descriptor.DType != DType.I8)
            {
                continue;
            }
            bool transformerLinear = key.StartsWith("decoder.transformer_blocks.", StringComparison.Ordinal)
                && (key.EndsWith(".attn.to_qkv.weight", StringComparison.Ordinal)
                    || key.EndsWith(".attn.to_out.weight", StringComparison.Ordinal)
                    || key.EndsWith(".ff.w1.weight", StringComparison.Ordinal)
                    || key.EndsWith(".ff.w2.weight", StringComparison.Ordinal));
            if (!transformerLinear)
            {
                issues.Add(Error("video.component.quant_scope_invalid",
                    $"Published int8 H3 video VAE quantization applies only to decoder transformer linears; got '{key}'.",
                    role));
                continue;
            }
            ValidateInt8Companions(descriptors, key[..^".weight".Length], descriptor, "video.component", role, issues);
        }
    }

    private static void ValidateAudioVaeStructure(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        List<VideoPlanIssue> issues, string role)
    {
        ValidateComponentShape(descriptors, "dec_in_proj.weight", role, issues, 2048, 32, 1);
        ValidateComponentShape(descriptors, "dec_in_proj.bias", role, issues, 2048);
        ValidateComponentShape(descriptors, "decoder.conv_pre.weight_v", role, issues, 1024, 2048, 7);
        ValidateComponentShape(descriptors, "decoder.conv_pre.weight_g", role, issues, 1024, 1, 1);
        ValidateComponentShape(descriptors, "decoder.conv_pre.bias", role, issues, 1024);
        int[] decoderChannels = [512, 256, 128, 64, 32, 16, 8];
        int decoderInput = 1024;
        for (int stage = 0; stage < decoderChannels.Length; stage++)
        {
            int output = decoderChannels[stage];
            int kernel = stage < 2 ? 9 : 4;
            string up = $"decoder.ups.{stage}.0";
            ValidateComponentShape(descriptors, up + ".weight_v", role, issues, decoderInput, output, kernel);
            ValidateComponentShape(descriptors, up + ".weight_g", role, issues, decoderInput, 1, 1);
            ValidateComponentShape(descriptors, up + ".bias", role, issues, output);
            for (int branch = 0; branch < 3; branch++)
            {
                int resblock = stage * 3 + branch;
                int resKernel = branch switch { 0 => 3, 1 => 7, _ => 11 };
                for (int layer = 0; layer < 3; layer++)
                {
                    string prefix = $"decoder.resblocks.{resblock}";
                    ValidateComponentShape(descriptors, $"{prefix}.convs1.{layer}.weight_v", role, issues,
                        output, output, resKernel);
                    ValidateComponentShape(descriptors, $"{prefix}.convs1.{layer}.weight_g", role, issues, output, 1, 1);
                    ValidateComponentShape(descriptors, $"{prefix}.convs1.{layer}.bias", role, issues, output);
                    ValidateComponentShape(descriptors, $"{prefix}.convs2.{layer}.weight_v", role, issues,
                        output, output, resKernel);
                    ValidateComponentShape(descriptors, $"{prefix}.convs2.{layer}.weight_g", role, issues, output, 1, 1);
                    ValidateComponentShape(descriptors, $"{prefix}.convs2.{layer}.bias", role, issues, output);
                    for (int activation = layer * 2; activation <= layer * 2 + 1; activation++)
                    {
                        ValidateComponentShape(descriptors, $"{prefix}.activations.{activation}.act.alpha", role,
                            issues, output);
                        ValidateComponentShape(descriptors, $"{prefix}.activations.{activation}.act.beta", role,
                            issues, output);
                    }
                }
            }
            decoderInput = output;
        }
        ValidateComponentIndexedFamily(descriptors.Keys, "decoder.ups.", 7, role, issues);
        ValidateComponentIndexedFamily(descriptors.Keys, "decoder.resblocks.", 21, role, issues);
        ValidateComponentShape(descriptors, "decoder.activation_post.act.alpha", role, issues, 8);
        ValidateComponentShape(descriptors, "decoder.activation_post.act.beta", role, issues, 8);
        ValidateComponentShape(descriptors, "decoder.conv_post.weight_v", role, issues, 1, 8, 7);
        ValidateComponentShape(descriptors, "decoder.conv_post.weight_g", role, issues, 1, 1, 1);

        ValidateComponentShape(descriptors, "encoder.block.0.weight_v", role, issues, 64, 1, 7);
        ValidateComponentShape(descriptors, "encoder.block.0.weight_g", role, issues, 64, 1, 1);
        ValidateComponentShape(descriptors, "encoder.block.0.bias", role, issues, 64);
        ValidateComponentShape(descriptors, "pre_block.attn.qkv.weight", role, issues, 6144, 2048);
        ValidateComponentShape(descriptors, "pre_block.proj.weight", role, issues, 32, 2048);
        ValidateComponentShape(descriptors, "mean_proj.weight", role, issues, 32, 32, 1);
        ValidateComponentShape(descriptors, "mean_proj.bias", role, issues, 32);
        if (descriptors.Values.Any(descriptor => descriptor.DType == DType.I8 || descriptor.DType == DType.U8))
        {
            issues.Add(Error("video.component.quant_unsupported",
                "No quantized MiniMax-H3 audio VAE artifact has been published or certified.", role));
        }
    }

    private static void ValidateTextEncoderStructure(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        List<VideoPlanIssue> issues, string role)
    {
        if (!descriptors.TryGetValue("model.embed_tokens.weight", out SafeTensorDescriptor? embedding)
            || embedding.Shape.Rank != 2 || embedding.Shape[0] != 151936 || embedding.Shape[1] != 5120)
        {
            issues.Add(Error("video.component.tensor_shape_invalid",
                "MiniMax-H3 text encoder embedding must be [151936,5120].", role));
        }
        else if (embedding.DType == DType.I8)
        {
            if (!descriptors.TryGetValue("model.embed_tokens.weight_scale", out SafeTensorDescriptor? scale)
                || scale.DType != DType.F32 || !ShapeEquals(scale, 151936, 1))
            {
                issues.Add(Error("video.component.quant_scale_invalid",
                    "The int8 H3 text embedding requires an F32 [151936,1] row scale.", role));
            }
        }
        for (int i = 0; i < 50; i++)
        {
            string layer = $"model.layers.{i}";
            ValidateComponentShape(descriptors, layer + ".input_layernorm.weight", role, issues, 5120);
            ValidateComponentShape(descriptors, layer + ".post_attention_layernorm.weight", role, issues, 5120);
            ValidateComponentShape(descriptors, layer + ".self_attn.q_norm.weight", role, issues, 128);
            ValidateComponentShape(descriptors, layer + ".self_attn.k_norm.weight", role, issues, 128);
            ValidateTextLinear(descriptors, layer + ".self_attn.q_proj", 8192, 5120, role, issues);
            ValidateTextLinear(descriptors, layer + ".self_attn.k_proj", 1024, 5120, role, issues);
            ValidateTextLinear(descriptors, layer + ".self_attn.v_proj", 1024, 5120, role, issues);
            ValidateTextLinear(descriptors, layer + ".self_attn.o_proj", 5120, 8192, role, issues);
            ValidateTextLinear(descriptors, layer + ".mlp.gate_proj", 25600, 5120, role, issues);
            ValidateTextLinear(descriptors, layer + ".mlp.up_proj", 25600, 5120, role, issues);
            ValidateTextLinear(descriptors, layer + ".mlp.down_proj", 5120, 25600, role, issues);
        }
        ValidateComponentIndexedFamily(descriptors.Keys, "model.layers.", 50, role, issues);
        ValidateComponentShape(descriptors, "visual.patch_embed.proj.weight", role, issues, 1152, 3, 2, 16, 16);
        ValidateComponentShape(descriptors, "visual.merger.norm.weight", role, issues, 1152);
        ValidateComponentShape(descriptors, "visual.merger.linear_fc1.weight", role, issues, 4608, 4608);
        ValidateComponentShape(descriptors, "visual.merger.linear_fc2.weight", role, issues, 5120, 4608);
    }

    private static void ValidateTextLinear(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string stem, long output, long input, string role, List<VideoPlanIssue> issues)
    {
        if (!descriptors.TryGetValue(stem + ".weight", out SafeTensorDescriptor? weight))
        {
            issues.Add(Error("video.component.tensor_missing", $"Component '{role}' is missing '{stem}.weight'.", role));
            return;
        }
        if (weight.DType != DType.U8)
        {
            if (!ShapeEquals(weight, output, input))
            {
                issues.Add(Error("video.component.tensor_shape_invalid",
                    $"Text tensor '{stem}.weight' must be [{output},{input}], got {weight.Shape}.", role));
            }
            return;
        }
        if (!ShapeEquals(weight, output, input / 2)
            || !descriptors.TryGetValue(stem + ".weight_scale", out SafeTensorDescriptor? blockScale)
            || blockScale.DType != DType.F8E4M3 || blockScale.Shape.Rank != 2
            || blockScale.Shape[0] < output || blockScale.Shape[1] != input / 16
            || !descriptors.TryGetValue(stem + ".weight_scale_2", out SafeTensorDescriptor? globalScale)
            || globalScale.DType != DType.F32 || ElementCount(globalScale) != 1
            || !descriptors.TryGetValue(stem + ".comfy_quant", out SafeTensorDescriptor? quant)
            || quant.DType != DType.U8 || quant.Shape.Rank != 1)
        {
            issues.Add(Error("video.component.nvfp4_invalid",
                $"NVFP4 text tensor '{stem}' has an invalid packed shape or companion set.", role));
        }
    }

    private static void ValidateComponentShape(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string key, string role, List<VideoPlanIssue> issues, params long[] expected)
    {
        if (!descriptors.TryGetValue(key, out SafeTensorDescriptor? descriptor))
        {
            issues.Add(Error("video.component.tensor_missing", $"Component '{role}' is missing '{key}'.", role));
        }
        else if (!ShapeEquals(descriptor, expected))
        {
            issues.Add(Error("video.component.tensor_shape_invalid",
                $"Component '{role}' tensor '{key}' must be [{string.Join(',', expected)}], got {descriptor.Shape}.",
                role));
        }
    }

    private static void ValidateComponentIndexedFamily(IEnumerable<string> keys, string prefix, int expectedCount,
        string role, List<VideoPlanIssue> issues)
    {
        HashSet<int> found = [];
        foreach (string key in keys)
        {
            if (!key.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            ReadOnlySpan<char> suffix = key.AsSpan(prefix.Length);
            int dot = suffix.IndexOf('.');
            if (dot > 0 && int.TryParse(suffix[..dot], NumberStyles.None, CultureInfo.InvariantCulture, out int index))
            {
                found.Add(index);
            }
        }
        int[] missing = Enumerable.Range(0, expectedCount).Where(index => !found.Contains(index)).ToArray();
        int[] extra = found.Where(index => index < 0 || index >= expectedCount).Order().ToArray();
        if (missing.Length > 0 || extra.Length > 0)
        {
            issues.Add(Error("video.component.block_count_invalid",
                $"Component '{role}' family '{prefix}' needs indices 0..{expectedCount - 1}; missing "
                + $"[{string.Join(',', missing)}], unexpected [{string.Join(',', extra)}].", role));
        }
    }

    private static VideoKnownArtifact? ConvertedPddArtifact(string hash,
        IReadOnlyDictionary<string, string> metadata, string? baseHash, List<VideoPlanIssue> issues)
    {
        if (!metadata.TryGetValue("hartsy.pdd.format", out string? format)
            || !string.Equals(format, "minimax_h3_pdd_hartsy_pruned_v1", StringComparison.Ordinal))
        {
            return null;
        }

        bool valid = true;
        if (!metadata.TryGetValue("hartsy.pdd.target_base_sha256", out string? targetHash)
            || !IsSha256(targetHash))
        {
            issues.Add(Error("video.pdd.target_hash_missing",
                "Converted PDD metadata must record a valid hartsy.pdd.target_base_sha256.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }
        else if (baseHash is null || !string.Equals(targetHash, baseHash, StringComparison.OrdinalIgnoreCase))
        {
            issues.Add(Error("video.pdd.target_hash_mismatch",
                $"Converted PDD adapter targets base SHA-256 {targetHash}, but the selected base is {baseHash ?? "unknown"}.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }

        foreach (string key in new[] { "hartsy.pdd.adapter_sha256", "hartsy.pdd.full_base_sha256" })
        {
            if (!metadata.TryGetValue(key, out string? sourceHash) || !IsSha256(sourceHash))
            {
                issues.Add(Error("video.pdd.provenance_hash_missing",
                    $"Converted PDD metadata must record a valid {key}.", nameof(VideoRequest.Loras)));
                valid = false;
            }
        }

        if (!metadata.TryGetValue("pdd_num_steps", out string? numStepsText)
            || !int.TryParse(numStepsText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int numSteps)
            || numSteps != MiniMaxH3PddSchedule.PublishedFineSteps)
        {
            issues.Add(Error("video.pdd.num_steps_invalid",
                $"Converted PDD adapters require pdd_num_steps={MiniMaxH3PddSchedule.PublishedFineSteps}.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }
        if (!metadata.TryGetValue("pdd_block_size", out string? blockSizeText)
            || !int.TryParse(blockSizeText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int blockSize)
            || blockSize != MiniMaxH3PddSchedule.PublishedBlockSize)
        {
            issues.Add(Error("video.pdd.block_size_invalid",
                $"Converted PDD adapters require pdd_block_size={MiniMaxH3PddSchedule.PublishedBlockSize}.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }
        if (!metadata.TryGetValue("hartsy.pdd.affine_residual", out string? residualText)
            || !double.TryParse(residualText, NumberStyles.Float, CultureInfo.InvariantCulture, out double residual)
            || !double.IsFinite(residual) || residual < 0.0 || residual > 1e-4)
        {
            issues.Add(Error("video.pdd.affine_residual_invalid",
                "Converted PDD metadata must record an affine residual no greater than 1e-4.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }

        VideoTaskFamily task = metadata.TryGetValue("hartsy.pdd.task", out string? taskText)
            ? NormalizePddTask(taskText)
            : VideoTaskFamily.Unknown;
        if (task is not (VideoTaskFamily.Fl2Va or VideoTaskFamily.Ref2Va))
        {
            issues.Add(Error("video.pdd.task_invalid",
                "Converted PDD metadata must bind hartsy.pdd.task to fl2va or ref2va.",
                nameof(VideoRequest.Loras)));
            valid = false;
        }
        if (!valid)
        {
            return null;
        }

        return new VideoKnownArtifact
        {
            Sha256 = hash,
            Id = $"minimax-h3-pdd-{task.ToString().ToLowerInvariant()}-rebased-{hash[..12]}",
            DisplayName = $"Locally rebased MiniMax-H3 PDD {task}",
            Role = VideoProfileArtifactRole.Adapter,
            Task = task,
            Acceleration = VideoAccelerationKind.Pdd,
            Steps = 8,
            FlowShift = 12f,
            AudioFlowShift = 3f,
            ReferenceSizing = VideoReferenceSizing.Native,
            ProvenanceUrl = "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs",
        };
    }

    private static VideoTaskFamily NormalizePddTask(string value)
    {
        string normalized = value.Trim().Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
        return normalized switch
        {
            "fl2va" => VideoTaskFamily.Fl2Va,
            "ref2va" => VideoTaskFamily.Ref2Va,
            _ => VideoTaskFamily.Unknown,
        };
    }

    private static bool IsSha256(string? value) => value is { Length: 64 }
        && value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');

    private static string BuildCacheIdentity(VideoModelProfile profile, Dictionary<string, string> hashes,
        Dictionary<string, string> paths)
    {
        StringBuilder builder = new StringBuilder();
        builder.Append("profile:").Append(profile.Id).Append(';');
        VideoDefaults defaults = profile.Defaults;
        builder.Append("contract:task=").Append((int)profile.Task)
            .Append(",acceleration=").Append((int)profile.Acceleration)
            .Append(",attention=").Append((int)profile.Attention)
            .Append(",features=").Append((long)profile.Features)
            .Append(",steps=").Append(defaults.Steps)
            .Append(",cfg=").Append(defaults.CfgScale.ToString("R", CultureInfo.InvariantCulture))
            .Append(",flow=").Append((defaults.FlowShift ?? float.NaN).ToString("R", CultureInfo.InvariantCulture))
            .Append(",audioFlow=").Append((defaults.AudioFlowShift ?? float.NaN)
                .ToString("R", CultureInfo.InvariantCulture))
            .Append(",sampler=").Append((defaults.Sampler ?? string.Empty).ToLowerInvariant())
            .Append(",scheduler=").Append((defaults.Scheduler ?? string.Empty).ToLowerInvariant())
            .Append(",width=").Append(defaults.Width?.ToString(CultureInfo.InvariantCulture) ?? "null")
            .Append(",height=").Append(defaults.Height?.ToString(CultureInfo.InvariantCulture) ?? "null")
            .Append(",referenceSizing=").Append((int)defaults.ReferenceSizing)
            .Append(",locked=").Append((int)defaults.LockedFields).Append(';');
        foreach (KeyValuePair<string, string> entry in hashes.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            builder.Append("sha:").Append(entry.Key).Append('=').Append(entry.Value).Append(';');
        }
        foreach (KeyValuePair<string, string> entry in paths.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            builder.Append("path:").Append(entry.Key).Append('=').Append(VideoArtifactPath.Identity(entry.Value))
                .Append(';');
        }
        return builder.ToString();
    }

    private static VideoPlanIssue Error(string code, string message, string? field = null) =>
        new VideoPlanIssue { Code = code, Severity = VideoPlanIssueSeverity.Error, Message = message, Field = field };

    private static VideoPlanIssue Warning(string code, string message, string? field = null) =>
        new VideoPlanIssue { Code = code, Severity = VideoPlanIssueSeverity.Warning, Message = message, Field = field };

    private sealed record HeaderSnapshot(IReadOnlyDictionary<string, SafeTensorDescriptor> Descriptors,
        IReadOnlyDictionary<string, string> Metadata);
}
