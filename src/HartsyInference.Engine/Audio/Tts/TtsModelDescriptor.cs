namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic speech path: how to turn a variant hint into a HuggingFace repo
/// (a constant when the pipeline resolves its own weights), and how to load it into an <see cref="ITtsRunner"/>.</summary>
internal sealed class TtsModelDescriptor
{
    /// <summary>Maps the request's variant hint to a HuggingFace repo id.</summary>
    internal required Func<string, string> ResolveRepo { get; init; }

    /// <summary>Loads the model (downloading on first use) into a uniform runner.</summary>
    internal required Func<string, CancellationToken, Task<ITtsRunner>> LoadAsync { get; init; }
}
