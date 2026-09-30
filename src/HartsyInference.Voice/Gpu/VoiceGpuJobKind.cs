namespace HartsyInference.Voice.Gpu;

/// <summary>What a <see cref="VoiceGpuWorker"/> job does, for logs and the order of shutdown.</summary>
internal enum VoiceGpuJobKind
{
    /// <summary>Speech recognition of one utterance.</summary>
    Transcribe,

    /// <summary>Synthesis of one sentence.</summary>
    Synthesize,

    /// <summary>A warm-up pass at load.</summary>
    Warm,

    /// <summary>Return of cached pool memory to the driver between turns; skipped when more work is queued behind it.</summary>
    Trim,

    /// <summary>The last job: runs its work (releasing the models on this thread) and ends the thread.</summary>
    Shutdown,
}
