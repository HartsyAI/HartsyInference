namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Structure read from a Kim_Vocal_2-style TFC-TDF-UNet graph: widths per level and the frequency size at the top.</summary>
/// <param name="InChannels">Spectrogram channels in and out (real/imag of left and right: 4).</param>
/// <param name="Widths">Channel width of each encoder level, then the bottleneck width last.</param>
/// <param name="DimF">Frequency bins at the top level.</param>
/// <param name="TfcLayers">3x3 conv layers per TFC block.</param>
public readonly record struct KimVocal2Geometry(int InChannels, int[] Widths, int DimF, int TfcLayers)
{
    /// <summary>Encoder levels (the bottleneck is not counted).</summary>
    public int Levels => Widths.Length - 1;
}
