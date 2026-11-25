namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>A video latent's frame count and spatial size, in latent pixels. Height and width are even: each 2x2 patch is one token.</summary>
public readonly record struct DuoGrid(int Frames, int Height, int Width)
{
    /// <summary>Video tokens per latent frame.</summary>
    public int TokensPerFrame => Height / 2 * (Width / 2);

    /// <summary>Video rows this grid packs into the sequence.</summary>
    public int VideoRows => Frames * TokensPerFrame;
}

/// <summary>One assistant block: its time on the 25 ms clock, its Intent row count, and its media. Frame starts are in milliseconds, one per latent frame.</summary>
public sealed record DuoBlockInput
{
    public required double StartMs { get; init; }

    /// <summary>Intent rows of this block's assistant body.</summary>
    public required int BodyTextRows { get; init; }

    public required DuoGrid Video { get; init; }

    /// <summary>Latent steps per audio channel; the audio packs two channels.</summary>
    public required int AudioSteps { get; init; }

    public required double[] FrameStartsMs { get; init; }
}

/// <summary>One conversation: the reference, the two shared Intent spans, the blocks, and which blocks are clean history or denoised now.</summary>
public sealed record DuoSampleInput
{
    public required int SampleId { get; init; }

    public required int SharedDescriptionRows { get; init; }

    public required int SharedCueRows { get; init; }

    /// <summary>The assistant portrait latent: one frame, video only.</summary>
    public required DuoGrid Reference { get; init; }

    public required IReadOnlyList<DuoBlockInput> Blocks { get; init; }

    /// <summary>Blocks <c>0 .. CleanBlocks-1</c> are committed clean history.</summary>
    public required int CleanBlocks { get; init; }

    /// <summary>Blocks whose media is denoised in this pass; each gets a noisy group.</summary>
    public required IReadOnlyList<int> NoisyBlocks { get; init; }
}
