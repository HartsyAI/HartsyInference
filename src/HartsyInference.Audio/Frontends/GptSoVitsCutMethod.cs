namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS text cut methods (upstream <c>text_segmentation_method.py</c>).</summary>
public enum GptSoVitsCutMethod
{
    /// <summary>No cut: the whole text is one segment.</summary>
    Cut0,

    /// <summary>Four sentences per segment (upstream's default in <c>TTS.run</c> and both web UIs).</summary>
    Cut1,

    /// <summary>Sentences gathered until a segment passes 50 characters.</summary>
    Cut2,

    /// <summary>Cut at each Chinese full stop.</summary>
    Cut3,

    /// <summary>Cut at each English full stop not between digits.</summary>
    Cut4,

    /// <summary>Cut at every punctuation mark (upstream's <c>api_v2</c> default).</summary>
    Cut5,
}
