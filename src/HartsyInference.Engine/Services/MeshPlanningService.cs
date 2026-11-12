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

    internal MeshPlanningService(MemoryEstimationService memory) => _memory = memory;

    /// <inheritdoc/>
    public IReadOnlyList<MeshModelDescriptor> DescribeAll() => [.. MeshModelIds.Described.Select(Describe)];

    /// <inheritdoc/>
    public MeshModelDescriptor Describe(string modelId)
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
        List<MeshPlanIssue> issues = [];
        try
        {
            MeshRequestValidator.Validate(id, request, MeshService.MatterAvailable(id));
        }
        catch (Exception ex) when (ex is NotSupportedException or ArgumentException or HartsyInferenceException)
        {
            issues.Add(Rejection(ex, descriptor));
        }
        AddIgnoredFieldWarnings(request, descriptor, issues);
        IReadOnlyList<MeshMissingArtifact> missing = MissingArtifacts(spec, issues);
        MemoryFit memory = _memory.AssessMesh(id, request, request.Vram);
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
            MissingArtifacts = missing,
            Memory = memory,
        });
    }

    private static string[] GatedRepos(CatalogEntry? catalog) =>
        catalog is null ? [] : [.. catalog.Assets.Where(asset => asset.Role.Contains("gated", StringComparison.OrdinalIgnoreCase)).Select(asset => asset.Repo).Distinct()];

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
        foreach (MeshFieldSpec field in descriptor.Fields.Values.Where(f => f.Access == MeshFieldAccess.Ignored && IsSet(request, f.Name)))
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

    /// <summary>The catalog files not at their catalog path. The loader may find a file elsewhere (a legacy folder), so these are warnings; a model with no checkpoint at all cannot run.</summary>
    private static IReadOnlyList<MeshMissingArtifact> MissingArtifacts(ModelSpec spec, List<MeshPlanIssue> issues)
    {
        if (spec.LocalPath is null)
        {
            issues.Add(new MeshPlanIssue
            {
                Code = "mesh.model.not_installed",
                Severity = MeshPlanIssueSeverity.Error,
                Message = $"No checkpoint was found for '{spec.Requested}'; fetch its catalog files or supply the model path.",
            });
        }
        if (spec.Catalog is null)
        {
            return [];
        }
        List<MeshMissingArtifact> missing = [];
        foreach (ModelAsset asset in ModelDownloader.MissingAssets(spec.Catalog, spec.LocalPath))
        {
            bool required = !asset.Role.Contains("optional", StringComparison.OrdinalIgnoreCase);
            string relative = Path.Combine(asset.TargetSubdir, asset.FileName).Replace('\\', '/');
            missing.Add(new MeshMissingArtifact(asset.Role, relative, required));
            issues.Add(new MeshPlanIssue
            {
                Code = "mesh.artifact.missing",
                Severity = MeshPlanIssueSeverity.Warning,
                Message = $"{asset.Role} ({relative}) is not at its catalog path.",
            });
        }
        return missing;
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
        _ => false,
    };

    /// <summary>What will run for each tunable field, and whether the caller or the model chose it.</summary>
    private static IReadOnlyDictionary<string, ResolvedParameter> Settings(MeshRequest request, MeshModelDescriptor descriptor)
    {
        Dictionary<string, ResolvedParameter> settings = new(StringComparer.Ordinal);
        foreach (MeshFieldSpec field in descriptor.Fields.Values.Where(f => f.Access == MeshFieldAccess.Used))
        {
            bool set = IsSet(request, field.Name);
            ResolvedParameter parameter = set
                ? new ResolvedParameter(Requested(request, field.Name), ParameterSource.UserOverride)
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
        _ => null,
    };
}
