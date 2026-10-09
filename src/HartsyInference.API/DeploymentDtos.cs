using System.Text.Json.Serialization;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API;

/// <summary>The body of <c>POST /admin/deployments</c>: a named model to load onto a device.</summary>
public sealed class DeployRequest
{
    [JsonPropertyName("deployment_id")] public string? DeploymentId { get; set; }
    [JsonPropertyName("model")] public string? Model { get; set; }

    /// <summary>The device to load onto (for example <c>cuda:0</c>). Null uses the engine's primary device; automatic placement is not offered yet.</summary>
    [JsonPropertyName("device")] public string? Device { get; set; }
}

/// <summary>One deployment as the admin routes report it.</summary>
public sealed class DeploymentDto
{
    [JsonPropertyName("deployment_id")] public required string DeploymentId { get; init; }
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("device")] public string? Device { get; init; }
    [JsonPropertyName("state")] public required string State { get; init; }
    [JsonPropertyName("problem")] public string? Problem { get; init; }

    public static DeploymentDto From(DeploymentStatus status) => new()
    {
        DeploymentId = status.DeploymentId,
        Model = status.Model,
        Device = status.Device,
        State = ToWire(status.State),
        Problem = status.Problem,
    };

    /// <summary>The state as its lowercase wire name: <c>ready</c>, <c>loading</c> and so on.</summary>
    public static string ToWire(DeploymentState state) => state.ToString().ToLowerInvariant();
}

/// <summary>The <c>/admin/deployments</c> listing.</summary>
public sealed class DeploymentListResponse
{
    [JsonPropertyName("deployments")] public required IReadOnlyList<DeploymentDto> Deployments { get; init; }
}

/// <summary>One deployment's capacity as <c>/admin/capacity</c> reports it. The KV figures are pages, and null on a model with no KV pool.</summary>
public sealed class CapacityDto
{
    [JsonPropertyName("deployment_id")] public required string DeploymentId { get; init; }
    [JsonPropertyName("state")] public required string State { get; init; }
    [JsonPropertyName("active")] public int Active { get; init; }
    [JsonPropertyName("queued")] public int Queued { get; init; }
    [JsonPropertyName("max_concurrent")] public int MaxConcurrent { get; init; }
    [JsonPropertyName("kv_pages_free")] public int? KvPagesFree { get; init; }
    [JsonPropertyName("kv_pages_total")] public int? KvPagesTotal { get; init; }

    public static CapacityDto For(DeploymentStatus status, DeploymentCapacity? capacity) => new()
    {
        DeploymentId = status.DeploymentId,
        State = DeploymentDto.ToWire(capacity?.State ?? status.State),
        Active = capacity?.Active ?? 0,
        Queued = capacity?.Queued ?? 0,
        MaxConcurrent = capacity?.MaxConcurrent ?? 0,
        KvPagesFree = capacity?.KvPagesFree,
        KvPagesTotal = capacity?.KvPagesTotal,
    };
}

/// <summary>The <c>hartsy.status</c> frame a streamed reply sends while it waits or prefills. OpenAI clients ignore named frames, so the standard stream is unaffected.</summary>
public sealed class HartsyStatusDto
{
    [JsonPropertyName("phase")] public required string Phase { get; init; }
    [JsonPropertyName("queue_position")] public int QueuePosition { get; init; }
    [JsonPropertyName("prefill_done")] public int PrefillDone { get; init; }
    [JsonPropertyName("prefill_total")] public int PrefillTotal { get; init; }

    public static HartsyStatusDto From(TextStatus status) => new()
    {
        Phase = status.Phase,
        QueuePosition = status.QueuePosition,
        PrefillDone = status.PrefillDone,
        PrefillTotal = status.PrefillTotal,
    };
}
