using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.Engine.Services;

/// <summary>MOSS-VL-Realtime in the text service. The model is a cross-attention VLM with its own loader, so it takes neither the GGUF
/// language-model path nor a scheduler: a request runs on the pipeline route through <see cref="MossVlRuntime"/>.</summary>
public sealed partial class TextService
{
    /// <summary>Loads a MOSS-VL GGUF or checkpoint folder onto <paramref name="slot"/>. Caller has already unloaded what the slot held.</summary>
    private void LoadMossVl(TextDeviceSlot slot, string deviceKey, string path, TextRequest request)
    {
        if (slot.ExtraStageBackends is not null)
            DisposeStageBackends(slot);
        string buildSelector = string.IsNullOrWhiteSpace(request.Device) ? PrimaryDeviceKey() : request.Device.Trim().ToLowerInvariant();
        IBackend backend = slot.Backend ??= CreateBackendFor(buildSelector);
        ApplyCacheWeightCastsOverride(slot, request, [backend]);
        bool cpu = backend is Core.Backends.Cpu.CpuBackend;
        // A GGUF dequantizes everything on the CPU (about 42 GB for this model); on a GPU the packed projections stay packed.
        if (File.Exists(path))
            EnsureRamHeadroomFor(path, dequantizesEverything: cpu);
        // Decode reads the packed blocks through the quantized GEMV; a widened copy of a quantized weight would only serve prefill.
        if (!cpu && backend.Capabilities.QuantizedTypes is not null)
            backend.CacheQuantizedWeightCasts = false;
        slot.MossVl = MossVlRuntime.Load(path, backend, lowVramQuant: !string.IsNullOrEmpty(request.LowVramQuant));
        slot.Scheduler = null;
        slot.PlannedPlacement = null;
        slot.PlacementRequested = null;
        slot.PreloadRedundantWeightSplitsApplied = null;
        slot.LoadedPath = path;
        Logs.Info($"[TextService] Loaded MOSS-VL '{Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar))}' "
            + $"({(slot.MossVl.Model.IsGguf ? "GGUF" : "checkpoint")}, decoder and vision tower) on {deviceKey}.");
    }

    /// <summary>Answers the last user message about its images on the slot's MOSS-VL model, streaming the reply.</summary>
    private static GenOutcome RunMossVl(TextDeviceSlot slot, TextRequest request, Action<TextChunk>? sink, CancellationToken cancel)
    {
        if (request.ForcedTokenIds is not null || request.ReturnTokenScores)
            throw new HartsyInferenceException("Teacher forcing and token scores are not served for MOSS-VL.");
        IReadOnlyList<ImageData> images = LastUserImages(request);
        if (images.Count == 0)
            throw new HartsyInferenceException("MOSS-VL-Realtime answers about images: attach at least one image to the user message (CLI: --image).");
        int maxTokens = request.MaxTokens > 0 ? request.MaxTokens : 512;
        MossVlGeneration reply = slot.MossVl!.Generate(VisionQuestion(request), images, maxTokens, BuildVisionSampling(request),
            sink is null ? null : delta => sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = delta }), cancel);
        return new GenOutcome(reply.Text, reply.HitTokenLimit ? StopReason.Length : StopReason.Stop, reply.PromptTokens, reply.CompletionTokens);
    }

    /// <summary>Every image of the most recent user message that has any, in order.</summary>
    private static IReadOnlyList<ImageData> LastUserImages(TextRequest request)
    {
        for (int i = request.Messages.Count - 1; i >= 0; i--)
        {
            TextMessage m = request.Messages[i];
            if (m.Role == TextRole.User && m.Images is { Count: > 0 })
                return m.Images;
        }
        return [];
    }
}
