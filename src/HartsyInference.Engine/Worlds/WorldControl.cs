namespace HartsyInference.Engine.Worlds;

/// <summary>A key a control scheme reads, with the label the viewer shows beside it.</summary>
public readonly record struct WorldControl(string Key, string Label);
