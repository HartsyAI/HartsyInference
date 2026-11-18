namespace HartsyInference.Core.Backends.Attention;

/// <summary>Every admitted row declined before changing visible state or output; a later call may succeed when its
/// workspace, capture state or engine configuration changes.</summary>
public sealed class AttentionRecoverableException(string message) : InvalidOperationException(message);
