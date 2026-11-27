namespace HartsyInference.Engine.Worlds;

/// <summary>Maps the keys a person holds to the action string a world session takes. A host reports key names as they are
/// pressed and released; a scheme declares the names it reads. Escape and Q are left to the host to stop a run.</summary>
public interface IWorldControlScheme
{
    /// <summary>Short name for the viewer title, for example "SANA-WM camera".</summary>
    string Name { get; }

    /// <summary>How the session takes actions.</summary>
    WorldControlCadence Cadence { get; }

    /// <summary>The keys the scheme reads, in the order the viewer's legend lists them.</summary>
    IReadOnlyList<WorldControl> Controls { get; }

    /// <summary>One line on what the controls do to the world, or an empty string.</summary>
    string Note { get; }

    /// <summary>The action for the keys held now. Key names compare case-insensitively; an empty set is the idle action.</summary>
    string ActionFor(IReadOnlyCollection<string> heldKeys);
}
