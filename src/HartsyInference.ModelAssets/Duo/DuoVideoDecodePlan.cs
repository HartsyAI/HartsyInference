namespace HartsyInference.ModelAssets.Duo;

/// <summary>One call of the native video decoder in the <c>duo_5_17</c> policy: which received latent steps form the window, how many
/// raw frames the decoder returns for it, and which of them are published. Pure bookkeeping; no VAE math happens here.</summary>
public readonly record struct DuoDecodeStep(
    /// <summary>The block this call publishes.</summary>
    int Block,
    /// <summary>The first received chunk the window reads.</summary>
    int FirstChunk,
    /// <summary>The last received chunk the window reads; its first two latents (or, at EOF, its last two) are the lookahead.</summary>
    int LastChunk,
    /// <summary>Latent steps the window feeds the decoder: whole chunks before the last, plus two from the last.</summary>
    int WindowLatents,
    /// <summary>Raw decoded frames the decoder returns for this window (7 latents to 22 frames, 12 to 39).</summary>
    int RawFrames,
    /// <summary>Published frame range within the raw frames, <c>[CropStart, CropEnd)</c>.</summary>
    int CropStart,
    int CropEnd);

/// <summary>The <c>duo_5_17</c> decode schedule: each received chunk of five latent steps is decoded with a two-latent lookahead, so block
/// <c>k</c> is published when chunk <c>k+1</c> arrives. The first chunk publishes nothing, and at EOF the last block is published from
/// the last two latents. Mirrors <c>VideoVAE.decode_block</c> and <c>finish_decode</c> in <c>duo/models/realization/modules/video_vae.py</c> @068e357,
/// so the frame and latent counts, crops and blocks here are the ones the native decoder is driven with.</summary>
public static class DuoVideoDecodePlan
{
    private static readonly DuoCodecSpec Spec = DuoCodecSpec.Get("duo_5_17");

    /// <summary>The call made when chunk <paramref name="chunk"/> (0-based) arrives. Null for the first chunk, which publishes nothing.</summary>
    public static DuoDecodeStep? OnChunk(int chunk)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunk);
        if (chunk == 0)
        {
            return null;
        }
        // Window: the two previous chunks (or the one that exists), then the current chunk's first two latents.
        int first = Math.Max(0, chunk - 2);
        int previousLatents = (chunk - first) * Spec.Latents;
        int latents = previousLatents + 2;
        return new DuoDecodeStep(
            Block: chunk - 1,
            FirstChunk: first,
            LastChunk: chunk,
            WindowLatents: latents,
            RawFrames: RawFramesFor(latents),
            CropStart: chunk == 1 ? 0 : 17,
            CropEnd: chunk == 1 ? 17 : 34);
    }

    /// <summary>The call made at EOF after <paramref name="received"/> chunks. Null when nothing was received, and publishes the last
    /// block from its own last two latents.</summary>
    public static DuoDecodeStep? AtEnd(int received)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(received);
        if (received == 0)
        {
            return null;
        }
        int last = received - 1;
        int first = Math.Max(0, received - 2);
        int latents = (last - first + 1) * Spec.Latents + 2;
        return new DuoDecodeStep(
            Block: received - 1,
            FirstChunk: first,
            LastChunk: last,
            WindowLatents: latents,
            RawFrames: RawFramesFor(latents),
            CropStart: received == 1 ? 0 : 17,
            CropEnd: received == 1 ? 17 : 34);
    }

    /// <summary>The raw frame count the native decoder returns for a window of <paramref name="latents"/> latent steps. Only the two
    /// window sizes the policy uses exist: 7 latents give 22 frames, 12 give 39.</summary>
    public static int RawFramesFor(int latents) => latents switch
    {
        7 => 22,
        12 => 39,
        _ => throw new ArgumentOutOfRangeException(nameof(latents), latents, "The duo_5_17 decoder takes a 7- or 12-latent window."),
    };
}
