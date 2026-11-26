namespace HartsyInference.ModelAssets.Duo;

/// <summary>One call of the native video decoder in the <c>duo_5_17</c> policy: the window it reads, its raw frame count, and the
/// published crop <c>[CropStart, CropEnd)</c>. <c>EndOfStream</c> marks the EOF call, whose lookahead is the last chunk's last two latents.
/// Pure bookkeeping; no VAE math happens here.</summary>
public readonly record struct DuoDecodeStep(int Block, int FirstChunk, int LastChunk, int WindowLatents, int RawFrames, int CropStart, int CropEnd, bool EndOfStream);

/// <summary>The <c>duo_5_17</c> decode schedule: each received chunk of five latent steps is decoded with a two-latent lookahead, so block
/// <c>k</c> is published when chunk <c>k+1</c> arrives. The first chunk publishes nothing, and at EOF the last block is published from
/// the last two latents. Mirrors <c>VideoVAE.decode_block</c> and <c>finish_decode</c> in <c>duo/models/realization/modules/video_vae.py</c> @068e357.</summary>
public static class DuoVideoDecodePlan
{
    private const int ChunkLatents = 5;

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
        int latents = (chunk - first) * ChunkLatents + 2;
        return Step(chunk - 1, first, chunk, latents, chunk == 1, endOfStream: false);
    }

    /// <summary>The call made at EOF after <paramref name="received"/> chunks. Null when nothing was received. It publishes the last
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
        int latents = (last - first + 1) * ChunkLatents + 2;
        return Step(received - 1, first, last, latents, received == 1, endOfStream: true);
    }

    /// <summary>The raw frame count the native decoder returns for a window of <paramref name="latents"/> latent steps. Only the two
    /// window sizes the policy uses exist: 7 latents give 22 frames, 12 give 39.</summary>
    public static int RawFramesFor(int latents) => latents switch
    {
        7 => 22,
        12 => 39,
        _ => throw new ArgumentOutOfRangeException(nameof(latents), latents, "The duo_5_17 decoder takes a 7- or 12-latent window."),
    };

    private static DuoDecodeStep Step(int block, int first, int last, int latents, bool firstBlock, bool endOfStream) =>
        new(block, first, last, latents, RawFramesFor(latents), firstBlock ? 0 : 17, firstBlock ? 17 : 34, endOfStream);
}
