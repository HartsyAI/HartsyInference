namespace HartsyInference.Vision.DepthPro2;

/// <summary>How <see cref="DepthPro2VideoStream"/> groups frames (upstream <c>VideoMode</c>).</summary>
public enum DepthPro2VideoMode
{
    /// <summary>Overlapping clips predicted jointly and cross-faded; a frame also sees later frames and is returned once its clip is done.</summary>
    Buffered,

    /// <summary>Each frame predicted as it arrives from itself and the frames before it, through a key/value cache.</summary>
    Unbuffered,
}
