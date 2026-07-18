using HartsyInference.Engine.Dispatch.Handlers;
using HartsyInference.Engine;

namespace HartsyInference.Engine.Dispatch;

/// <summary>Registry mapping each <see cref="Modality"/> to its handler. Handlers are added here as they are wired;
/// unregistered modalities surface a clear "not yet available" error instead of a crash.</summary>
public sealed class ModalityDispatch
{
    private readonly Dictionary<Modality, IModalityHandler> _handlers = new();

    /// <summary>Creates the dispatch with all currently-wired handlers registered.</summary>
    public ModalityDispatch()
    {
        Register(new TextHandler());
        Register(new ImageHandler());
        Register(new TranscribeHandler());
        Register(new SpeechHandler());
        Register(new ThreeDHandler());
        Register(new VisionHandler());
        Register(new MusicHandler());
        Register(new VideoHandler());
        Register(new InteractiveHandler());
    }

    /// <summary>Registers (or replaces) the handler for its modality.</summary>
    public void Register(IModalityHandler handler) => _handlers[handler.Modality] = handler;

    /// <summary>Whether a handler is wired for <paramref name="modality"/>.</summary>
    public bool IsSupported(Modality modality) => _handlers.ContainsKey(modality);

    /// <summary>The modalities that currently have a handler, in display order.</summary>
    public IReadOnlyList<Modality> SupportedModalities => Modalities.All.Where(_handlers.ContainsKey).ToList();

    /// <summary>Gets the handler for <paramref name="modality"/>, or throws a clear error when it is not wired yet.</summary>
    public IModalityHandler Get(Modality modality)
    {
        if (_handlers.TryGetValue(modality, out IModalityHandler? handler))
            return handler;
        throw new NotSupportedException(
            $"The '{Modalities.ToCliName(modality)}' modality is not wired into the CLI yet. Wired so far: " +
            $"{string.Join(", ", SupportedModalities.Select(Modalities.ToCliName))}.");
    }
}
