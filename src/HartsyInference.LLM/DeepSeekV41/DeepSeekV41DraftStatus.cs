namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Whether a checkpoint carries a usable draft (<c>mtp.*</c>) stack.</summary>
public enum DeepSeekV41DraftStatus
{
    /// <summary>No <c>mtp.*</c> tensors at all (a producer stripped them).</summary>
    Absent,

    /// <summary>Every routed expert of every draft layer has all its tensors.</summary>
    Complete,

    /// <summary>Some routed experts of a draft layer lack tensors; the draft cannot run.</summary>
    Incomplete,
}
