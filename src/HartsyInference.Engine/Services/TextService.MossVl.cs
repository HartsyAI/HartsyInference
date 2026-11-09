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
        string[] shardDevices = ResolveShardDevices(deviceKey);
        if (shardDevices.Length >= 2)
        {
            // The cross-attention decoder has no layer split. A composite key passed whole to CreateBackendFor would parse its
            // ordinal from "0+cuda:1" and silently land on ordinal 0, so build on the first listed device and say so.
            Logs.Warning($"[TextService] '{deviceKey}' requests a {shardDevices.Length}-way split, but MOSS-VL loads on one device. "
                + $"Loading on '{shardDevices[0]}' only; the rest of the device list is ignored.");
            deviceKey = shardDevices[0];
            buildSelector = shardDevices[0];
        }
        IBackend backend = slot.Backend ??= CreateBackendFor(buildSelector);
        ApplyCacheWeightCastsOverride(slot, request, [backend]);
        bool cpu = backend is Core.Backends.Cpu.CpuBackend;
        // A GGUF dequantizes everything on the CPU (about 42 GB for this model); on a GPU the packed projections stay packed.
        if (File.Exists(path))
            EnsureRamHeadroomFor(path, dequantizesEverything: cpu);
        else if (cpu)
            EnsureRamHeadroomForMossVlFolderOnCpu(path, ReadAvailableMemoryKb());
        // Decode reads the packed blocks through the quantized GEMV; a widened copy of a quantized weight would only serve prefill.
        if (!cpu && backend.Capabilities.QuantizedTypes is not null)
            backend.CacheQuantizedWeightCasts = false;
        try
        {
            slot.MossVl = MossVlRuntime.Load(path, backend, lowVramQuant: !string.IsNullOrEmpty(request.LowVramQuant));
        }
        catch
        {
            // A preload that ran out of memory part-way leaves weights resident in the backend's cache. Disposing the host tensors
            // does not release them, and with slot.MossVl unset the next UnloadSlot would skip the device cleanup.
            try { backend.FreeAllDeviceMemory(); }
            catch (Exception freeError) { Logs.Warning($"[TextService] Freeing device memory after a failed MOSS-VL load failed: {freeError.Message}"); }
            throw;
        }
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
        // The model answers one user turn about its images; a system prompt and earlier turns are not replayed into the prompt.
        if (request.Messages.Count > 1 && slot.LoggedSettingMismatches.Add("MossVlHistory"))
            Logs.Info("[TextService] MOSS-VL answers only the last user message and its images; the system prompt and earlier turns are not used.");
        MossVlGeneration reply = slot.MossVl!.Generate(VisionQuestion(request), images, maxTokens, BuildVisionSampling(request),
            sink is null ? null : delta => sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = delta }), cancel);
        return new GenOutcome(reply.Text, reply.HitTokenLimit ? StopReason.Length : StopReason.Stop, reply.PromptTokens, reply.CompletionTokens);
    }

    /// <summary>Refuses a MOSS-VL checkpoint folder on the CPU when the host cannot hold it widened: the CPU path keeps every projection as
    /// F32, twice the BF16 shards (about 42 GB for the released model), plus a fixed margin for activations and the mapped shards' pages.</summary>
    /// <param name="availableKb">Free host RAM in KiB; zero or less means unknown, and nothing is refused.</param>
    internal static void EnsureRamHeadroomForMossVlFolderOnCpu(string folder, long availableKb)
    {
        if (availableKb <= 0 || !Directory.Exists(folder))
            return;
        long shardBytes = Directory.EnumerateFiles(folder, "*.safetensors").Sum(f => new FileInfo(f).Length);
        const long MarginBytes = 4L << 30;
        long requiredBytes = 2 * shardBytes + MarginBytes;
        long availableBytes = availableKb * 1024;
        if (availableBytes < requiredBytes)
            throw new HartsyInferenceException(
                $"Not enough free host RAM to load MOSS-VL '{Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar))}' on the CPU: "
                + $"{availableBytes / (double)(1L << 30):0.0} GB free, need about {requiredBytes / (double)(1L << 30):0.0} GB "
                + $"(the {shardBytes / (double)(1L << 30):0.0} GB of shards widened to F32, plus a margin). Use a GPU device or the GGUF.");
    }

    /// <summary>Notes once per slot that a kept MOSS-VL model was loaded with a different low-VRAM setting than this request asks for.</summary>
    private static void LogMossVlLowVramMismatch(TextDeviceSlot slot, string deviceKey, TextRequest request)
    {
        if (slot.MossVl is not { } runtime || request.LowVramQuant is null)
            return;
        LogLoadTimeSettingMismatch(slot, deviceKey, "LowVramQuant", request.LowVramQuant.Length > 0, runtime.LowVramQuant);
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
