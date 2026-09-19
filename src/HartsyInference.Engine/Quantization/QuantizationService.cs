using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Planning;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.Engine.Quantization;

/// <summary>Quantizes a checkpoint and, when the SOURCE is a video build the engine recognizes, binds the OUTPUT to
/// the same execution semantics with a profile sidecar.
/// <para>Without that, quantizing an H3-class checkpoint silently downgrades it: video planning is bound to an
/// exact file hash, so a file we produced ourselves is a stranger and plans as <c>UnknownBaseProfile</c> — the
/// task, acceleration and step count the source declared are simply gone, with nothing to say so. The quantizer
/// itself cannot fix that; it does not know what a video profile is. This does.</para></summary>
public static class QuantizationService
{
    /// <summary>Runs <paramref name="job"/> and writes <c>&lt;output&gt;.hartsy-video-profile.json</c> beside the
    /// result when the source's hash resolves to a known video artifact.</summary>
    /// <returns>The quantizer's report and the sidecar path, or null when the source was not a recognized video
    /// build — the ordinary case for an image checkpoint, and not an error.</returns>
    public static async Task<(QuantizationReport Report, string? SidecarPath)> QuantizeAsync(
        QuantizationJob job, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(job);
        // Hashed BEFORE the write, so a job whose output lands beside its source still measured the real source,
        // and a quantize that throws leaves nothing half-bound.
        string sourceHash = await VideoCheckpointHashCache.GetSha256Async(job.SourcePath, cancel).ConfigureAwait(false);
        bool known = VideoProfileManifest.TryGetByHash(sourceHash, out VideoKnownArtifact? source);

        QuantizationReport report = CheckpointQuantizer.Quantize(job, cancel);
        if (!known || source is null)
        {
            Logs.Info($"[Quantize] Source {sourceHash[..12]} is not a known video artifact — no profile sidecar. "
                + "An image checkpoint, or a video build nobody has verified yet.");
            return (report, null);
        }

        string outputHash = await VideoCheckpointHashCache.GetSha256Async(job.OutputPath, cancel).ConfigureAwait(false);
        VideoProfileSidecar sidecar = new()
        {
            Sha256 = outputHash,
            ProfileId = source.Id + "-requant",
            DisplayName = source.DisplayName + " (HartsyInference requantized)",
            Task = source.Task,
            Acceleration = source.Acceleration,
            Attention = source.Attention,
            // Where the artifact declares nothing the base recipe supplies it, and the sidecar's own defaults are
            // that same base — so a null becomes the default rather than zero, which would lock a real value.
            Steps = source.Steps ?? 0,
            FlowShift = source.FlowShift ?? 12f,
            AudioFlowShift = source.AudioFlowShift ?? 3f,
            Width = source.Width,
            Height = source.Height,
            ReferenceSizing = source.ReferenceSizing,
            ProvenanceUrl = source.ProvenanceUrl,
        };
        string sidecarPath = job.OutputPath + ".hartsy-video-profile.json";
        await using (FileStream stream = File.Create(sidecarPath))
        {
            await JsonSerializer.SerializeAsync(stream, sidecar,
                VideoPlanningJsonContext.Default.VideoProfileSidecar, cancel).ConfigureAwait(false);
        }
        Logs.Info($"[Quantize] Bound the output to '{sidecar.ProfileId}' ({source.Task}) via {sidecarPath} — "
            + "planning reads it by hash, so the requantized build keeps the source's semantics.");
        return (report, sidecarPath);
    }
}
