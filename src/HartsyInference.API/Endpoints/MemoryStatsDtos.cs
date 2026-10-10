using System.Text.Json.Serialization;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>GET <c>/admin/memory</c>: the free host RAM, and the residency plan of each loaded text model.</summary>
public sealed record MemoryStatsResponse
{
    /// <summary>Free host RAM in bytes; null when the host does not report it.</summary>
    [JsonPropertyName("host_available_bytes")] public required long? HostAvailableBytes { get; init; }

    [JsonPropertyName("models")] public required IReadOnlyList<LoadedModelMemoryDto> Models { get; init; }

    /// <summary>Where each loaded GGUF text model was placed: one GPU, a layer split, or expert offload.</summary>
    [JsonPropertyName("placements")] public IReadOnlyList<LoadedModelPlacementDto> Placements { get; init; } = [];
}

/// <summary>One loaded text model's placement, as the planner decided it at load.</summary>
public sealed record LoadedModelPlacementDto
{
    [JsonPropertyName("device_key")] public required string DeviceKey { get; init; }

    [JsonPropertyName("model_path")] public required string ModelPath { get; init; }

    /// <summary><c>gpu</c>, <c>split</c> or <c>offload</c>.</summary>
    [JsonPropertyName("mode")] public required string Mode { get; init; }

    /// <summary>The devices the model runs on, in stage order.</summary>
    [JsonPropertyName("devices")] public required IReadOnlyList<string> Devices { get; init; }

    /// <summary>Device bytes the expert cache may hold; zero unless the mode is <c>offload</c>.</summary>
    [JsonPropertyName("expert_budget_bytes")] public required long ExpertBudgetBytes { get; init; }

    [JsonPropertyName("required_bytes")] public required long RequiredBytes { get; init; }

    [JsonPropertyName("available_bytes")] public required long AvailableBytes { get; init; }

    [JsonPropertyName("reason")] public required string Reason { get; init; }

    internal static LoadedModelPlacementDto For(LoadedModelPlacement loaded) => new()
    {
        DeviceKey = loaded.DeviceKey,
        ModelPath = loaded.ModelPath,
        Mode = loaded.Placement.Mode.ToString().ToLowerInvariant(),
        Devices = loaded.Placement.Devices,
        ExpertBudgetBytes = loaded.Placement.ExpertBudgetBytes,
        RequiredBytes = loaded.Placement.RequiredBytes,
        AvailableBytes = loaded.Placement.AvailableBytes,
        Reason = loaded.Placement.Reason,
    };
}

/// <summary>One loaded text model's host residency. Expert-cache and Engram counters are not reported on the CPU host path, and the note says so instead of returning zeros.</summary>
public sealed record LoadedModelMemoryDto
{
    [JsonPropertyName("device_key")] public required string DeviceKey { get; init; }

    [JsonPropertyName("model_path")] public required string ModelPath { get; init; }

    [JsonPropertyName("verdict")] public required string Verdict { get; init; }

    [JsonPropertyName("reason")] public required string Reason { get; init; }

    [JsonPropertyName("available_bytes")] public required long AvailableBytes { get; init; }

    [JsonPropertyName("working_set_bytes")] public required long WorkingSetBytes { get; init; }

    [JsonPropertyName("headroom_bytes")] public required long HeadroomBytes { get; init; }

    [JsonPropertyName("mapped_bytes")] public required long MappedBytes { get; init; }

    [JsonPropertyName("reserved_by_account")] public required IReadOnlyDictionary<string, long> ReservedByAccount { get; init; }

    [JsonPropertyName("components")] public required IReadOnlyList<ComponentMemoryDto> Components { get; init; }

    [JsonPropertyName("stats_note")] public string StatsNote { get; init; } = "Expert-cache and Engram counters are not reported on the CPU host path.";

    public static LoadedModelMemoryDto For(LoadedModelResidency loaded)
    {
        ResidencyPlan plan = loaded.Plan;
        return new LoadedModelMemoryDto
        {
            DeviceKey = loaded.DeviceKey,
            ModelPath = loaded.ModelPath,
            Verdict = plan.Verdict.ToString(),
            Reason = plan.Reason,
            AvailableBytes = plan.AvailableBytes,
            WorkingSetBytes = plan.WorkingSetBytes,
            HeadroomBytes = plan.HeadroomBytes,
            MappedBytes = plan.MappedBytes,
            ReservedByAccount = plan.ReservedByAccount.ToDictionary(entry => entry.Key.ToString(), entry => entry.Value),
            Components = [.. plan.Components.Select(component => new ComponentMemoryDto(component.Component.ToString(), component.Mode.ToString(), component.Device, component.Bytes))],
        };
    }
}

/// <summary>One component of a loaded model: where it runs and its stored bytes.</summary>
public sealed record ComponentMemoryDto(
    [property: JsonPropertyName("component")] string Component,
    [property: JsonPropertyName("mode")] string Mode,
    [property: JsonPropertyName("device")] string Device,
    [property: JsonPropertyName("bytes")] long Bytes);
