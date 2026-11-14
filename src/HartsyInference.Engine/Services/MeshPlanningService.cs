using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Answers <see cref="IMeshPlanningService"/> from the model profiles, the request validator, the catalog and the device, without loading weights.</summary>
public sealed class MeshPlanningService : IMeshPlanningService
{
    private readonly MemoryEstimationService _memory;
    private readonly IMeshAssetService _assets;

    internal MeshPlanningService(MemoryEstimationService memory, IMeshAssetService assets)
    {
        _memory = memory;
        _assets = assets;
    }

    /// <inheritdoc/>
    public IReadOnlyList<MeshModelDescriptor> DescribeAll() => [.. MeshModelIds.Described.Select(Describe)];

    /// <inheritdoc/>
    public MeshModelDescriptor Describe(string modelId) => DescriptorOf(modelId);

    /// <summary>The descriptor of <paramref name="modelId"/>; the service holds no state, so a generation reads it too.</summary>
    internal static MeshModelDescriptor DescriptorOf(string modelId)
    {
        ArgumentNullException.ThrowIfNull(modelId);
        string id = modelId.ToLowerInvariant();
        MeshModelProfile profile = MeshModelProfiles.Find(id)
            ?? throw new NotSupportedException($"3D model '{modelId}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        CatalogEntry? catalog = ModelCatalog.Find(id);
        return new MeshModelDescriptor
        {
            Id = profile.Id,
            DisplayName = catalog?.DisplayName ?? profile.Id,
            Input = profile.Input,
            Outputs = profile.Outputs,
            Status = catalog?.Status ?? ModelStatus.Structural,
            Fields = profile.Fields.ToDictionary(field => field.Name, StringComparer.Ordinal),
            Presets = profile.Presets,
            Licence = profile.Licence with { GatedRepos = GatedRepos(catalog) },
            Cost = profile.Cost,
            Note = profile.Note,
        };
    }

    /// <inheritdoc/>
    public Task<MeshPlan> PlanAsync(ModelSpec spec, MeshRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        string id = MeshService.IdOf(spec);
        MeshModelDescriptor descriptor = Describe(id);
        MeshRequest resolved = MeshQualityPresets.Resolve(id, request);
        List<MeshPlanIssue> issues = [];
        try
        {
            MeshRequestValidator.Validate(id, resolved, MeshService.MatterAvailable());
        }
        catch (Exception ex) when (ex is NotSupportedException or ArgumentException or HartsyInferenceException)
        {
            issues.Add(Rejection(ex, descriptor));
        }
        AddIgnoredFieldWarnings(request, descriptor, issues);
        MeshAssetStatus assets = _assets.GetStatus(spec);
        AddAssetIssues(spec, assets, issues);
        MemoryFit memory = _memory.AssessMesh(id, resolved, request.Vram);
        if (memory.Verdict == MemoryFitVerdict.Infeasible)
        {
            issues.Add(new MeshPlanIssue { Code = "mesh.memory.infeasible", Severity = MeshPlanIssueSeverity.Error, Message = memory.Reason });
        }
        return Task.FromResult(new MeshPlan
        {
            Model = spec,
            Descriptor = descriptor,
            Settings = Settings(request, descriptor),
            Issues = issues,
            Assets = assets,
            Memory = memory,
        });
    }

    private static string[] GatedRepos(CatalogEntry? catalog) =>
        catalog is null ? [] : [.. catalog.Assets.Where(MeshAssetService.IsGated).Select(asset => asset.Repo).Distinct()];

    private static MeshPlanIssue Rejection(Exception error, MeshModelDescriptor descriptor)
    {
        string code = error switch
        {
            NotSupportedException => "mesh.request.unsupported",
            HartsyInferenceException => "mesh.request.rejected",
            _ => "mesh.request.invalid",
        };
        string? field = MeshFieldTag.FieldOf(error)
            ?? descriptor.Fields.Keys.OrderByDescending(name => name.Length).FirstOrDefault(name => error.Message.Contains(name, StringComparison.Ordinal));
        return new MeshPlanIssue { Code = code, Severity = MeshPlanIssueSeverity.Error, Message = error.Message, Field = field };
    }

    private static void AddIgnoredFieldWarnings(MeshRequest request, MeshModelDescriptor descriptor, List<MeshPlanIssue> issues)
    {
        // A field that already has an error is not also reported as ignored: the error says what is wrong with it.
        foreach (MeshFieldSpec field in descriptor.Fields.Values.Where(f => f.Access == MeshFieldAccess.Ignored && IsSet(request, f.Name) && !issues.Any(i => i.Field == f.Name && i.Severity == MeshPlanIssueSeverity.Error)))
        {
            issues.Add(new MeshPlanIssue
            {
                Code = "mesh.field.ignored",
                Severity = MeshPlanIssueSeverity.Warning,
                Message = $"'{descriptor.Id}' ignores {field.Name}. {field.Note}".TrimEnd(),
                Field = field.Name,
            });
        }
    }

    /// <summary>One warning per absent file (the loader may find a model path the catalog does not know, so an absent file does not block), with what to do when the engine cannot fetch it; an error when no checkpoint resolved at all.</summary>
    private static void AddAssetIssues(ModelSpec spec, MeshAssetStatus status, List<MeshPlanIssue> issues)
    {
        if (spec.LocalPath is null)
        {
            issues.Add(new MeshPlanIssue
            {
                Code = "mesh.model.not_installed",
                Severity = MeshPlanIssueSeverity.Error,
                Message = status.ModelBlocker is MeshAssetBlocker whole
                    ? $"No checkpoint was found for '{spec.Requested}'. {whole.Message}"
                    : $"No checkpoint was found for '{spec.Requested}'; fetch its catalog files or supply the model path.",
            });
        }
        foreach (MeshAssetInfo asset in status.Missing)
        {
            issues.Add(new MeshPlanIssue
            {
                Code = asset.Blocker is null ? "mesh.artifact.missing" : "mesh.artifact.blocked",
                Severity = MeshPlanIssueSeverity.Warning,
                Message = asset.Instructions ?? $"{asset.Role} ({asset.RelativePath}) is not installed; fetch the model's files.",
            });
        }
    }

    private static bool IsSet(MeshRequest request, string name) => name switch
    {
        nameof(MeshRequest.Steps) => request.Steps != 0,
        nameof(MeshRequest.GridResolution) => request.GridResolution != 0,
        nameof(MeshRequest.Seed) => request.Seed >= 0,
        nameof(MeshRequest.FieldOfViewRadians) => request.FieldOfViewRadians is not null,
        nameof(MeshRequest.Samples) => request.Samples != 1,
        nameof(MeshRequest.Colors) => request.Colors is not null,
        nameof(MeshRequest.Chains) => request.Chains is not null,
        nameof(MeshRequest.Rigged) => request.Rigged,
        nameof(MeshRequest.Texture) => request.Texture,
        nameof(MeshRequest.TextureSize) => request.TextureSize != 0,
        nameof(MeshRequest.Preprocess) => request.Preprocess != MeshPreprocess.None,
        nameof(MeshRequest.PreviewTriangles) => request.PreviewTriangles != 0,
        nameof(MeshRequest.Frame) => request.Frame != MeshFrame.Native,
        nameof(MeshRequest.Mesh) => request.Mesh,
        nameof(MeshRequest.Control) => request.Control is not null,
        nameof(MeshRequest.Quality) => request.Quality != MeshQuality.Default,
        _ => false,
    };

    /// <summary>What will run for each tunable field, and whether the caller or the model chose it.</summary>
    internal static IReadOnlyDictionary<string, ResolvedParameter> Settings(MeshRequest request, MeshModelDescriptor descriptor)
    {
        Dictionary<string, ResolvedParameter> settings = new(StringComparer.Ordinal);
        foreach (MeshFieldSpec field in descriptor.Fields.Values.Where(f => f.Access == MeshFieldAccess.Used))
        {
            bool set = IsSet(request, field.Name);
            int? preset = set ? null : MeshQualityPresets.Supplied(descriptor.Presets.GetValueOrDefault(request.Quality), request, field.Name);
            // A preset value equal to the model's own default changes nothing, so the plan does not credit the preset with it.
            ResolvedParameter parameter = set
                ? new ResolvedParameter(Requested(request, field.Name), ParameterSource.UserOverride)
                : preset is int value && value.ToString(System.Globalization.CultureInfo.InvariantCulture) != field.Default
                    ? new ResolvedParameter(value.ToString(System.Globalization.CultureInfo.InvariantCulture), ParameterSource.ProfileDefault, $"the {request.Quality} preset")
                    : new ResolvedParameter(field.Default, ParameterSource.RecipeDefault);
            settings[field.Name] = parameter;
        }
        return settings;
    }

    private static string? Requested(MeshRequest request, string name) => name switch
    {
        nameof(MeshRequest.Steps) => request.Steps.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.GridResolution) => request.GridResolution.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.Seed) => request.Seed.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.FieldOfViewRadians) => request.FieldOfViewRadians?.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.Samples) => request.Samples.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.Colors) => request.Colors?.ToString().ToLowerInvariant(),
        nameof(MeshRequest.Chains) => request.Chains?.ToString().ToLowerInvariant(),
        nameof(MeshRequest.Rigged) => request.Rigged.ToString().ToLowerInvariant(),
        nameof(MeshRequest.Texture) => request.Texture.ToString().ToLowerInvariant(),
        nameof(MeshRequest.TextureSize) => request.TextureSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.Preprocess) => request.Preprocess.ToString(),
        nameof(MeshRequest.PreviewTriangles) => request.PreviewTriangles.ToString(System.Globalization.CultureInfo.InvariantCulture),
        nameof(MeshRequest.Frame) => request.Frame.ToString(),
        nameof(MeshRequest.Mesh) => request.Mesh.ToString().ToLowerInvariant(),
        nameof(MeshRequest.Control) => request.Control?.Kind.ToString(),
        nameof(MeshRequest.Quality) => request.Quality.ToString(),
        _ => null,
    };
}
