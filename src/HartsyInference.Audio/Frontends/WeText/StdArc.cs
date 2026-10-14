namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>An OpenFst <c>StdArc</c>: input and output labels (0 is epsilon), a tropical weight and the target
/// state.</summary>
public readonly record struct StdArc(int ILabel, int OLabel, float Weight, int NextState);
