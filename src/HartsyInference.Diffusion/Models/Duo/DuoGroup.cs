namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>What a packed-sequence group holds. The visibility rules in <see cref="DuoMask"/> are written against these kinds.</summary>
public enum DuoGroupKind
{
    /// <summary>The assistant portrait latent, one frame, shown to every row of its conversation.</summary>
    Reference,

    /// <summary>Intent rows shared by a conversation: the assistant description and the speech cue.</summary>
    Shared,

    /// <summary>One block's Intent rows.</summary>
    Text,

    /// <summary>A generated block kept as a clean history copy.</summary>
    Clean,

    /// <summary>The block being denoised.</summary>
    Noisy,

    /// <summary>Rows that round the packed length up to a multiple of 128.</summary>
    Padding,
}

/// <summary>The modality of a media group; text groups carry <see cref="None"/>.</summary>
public enum DuoModality
{
    None,
    Video,
    Audio,
}

/// <summary>A contiguous run of packed rows that the attention mask treats as one unit. <paramref name="Block"/> is -1 for groups not tied to a block.</summary>
public readonly record struct DuoGroup(int Sample, int Block, DuoGroupKind Kind, DuoModality Modality, int Start, int End)
{
    public int Length => End - Start;
}
