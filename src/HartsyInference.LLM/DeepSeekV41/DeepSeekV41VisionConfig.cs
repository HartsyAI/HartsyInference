using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Vision tower dimensions from the config, for the checkpoint's <c>vision.*</c> and <c>aligner.*</c> tensors.</summary>
/// <remarks><c>RopeTheta</c> is the base of the tower's 2D rotary; upstream defaults it to 10000 and a config may omit it.</remarks>
public sealed record DeepSeekV41VisionConfig(int NumLayers, int HiddenSize, int NumHeads, int IntermediateSize,
    int PatchSize, int DownsampleRatio, double RopeTheta = DeepSeekV41VisionConfig.DefaultRopeTheta)
{
    /// <summary>Upstream's <c>vision_rope_theta</c> default.</summary>
    public const double DefaultRopeTheta = 10000.0;

    /// <summary>Per-head width of the tower's attention.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Values in one flattened patch: three channels of <c>PatchSize x PatchSize</c> pixels.</summary>
    public int PatchInputDim => 3 * PatchSize * PatchSize;

    /// <summary>Width of one aligner input row: a <c>DownsampleRatio x DownsampleRatio</c> block of tower features.</summary>
    public int AlignerInputDim => HiddenSize * DownsampleRatio * DownsampleRatio;

    /// <summary>Throws unless the dimensions describe a runnable tower.</summary>
    /// <exception cref="HartsyInferenceException">A size is not positive, the heads do not divide the width, or the rotary width is not even.</exception>
    public void Validate()
    {
        if (NumLayers < 1 || HiddenSize < 1 || NumHeads < 1 || IntermediateSize < 1 || PatchSize < 1 || DownsampleRatio < 1)
            throw new HartsyInferenceException($"DeepSeek-V4.1 vision config needs positive sizes, got {this}.");
        if (HiddenSize % NumHeads != 0)
            throw new HartsyInferenceException($"DeepSeek-V4.1 vision hidden_size {HiddenSize} does not divide into {NumHeads} heads.");
        // the rotary spans half a head, as pairs of angles for the row and the column, so the head width must be a multiple of four
        if (HeadDim % 4 != 0)
            throw new HartsyInferenceException($"DeepSeek-V4.1 vision head width {HeadDim} must be a multiple of 4 for the 2D rotary.");
        if (!(RopeTheta > 0.0) || !double.IsFinite(RopeTheta))
            throw new HartsyInferenceException($"DeepSeek-V4.1 vision rope_theta must be a positive finite number, got {RopeTheta}.");
    }
}
