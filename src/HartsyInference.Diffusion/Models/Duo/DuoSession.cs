using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The media and text one Duo conversation run needs: the shared Intent rows, the reference latent rows, and for each block its body
/// Intent rows and the initial noise of its video and audio.</summary>
public sealed record DuoRun(float[] SharedText, float[] ReferenceVideo, IReadOnlyList<float[]> BodyText, IReadOnlyList<float[]> NoiseVideo,
    IReadOnlyList<float[]> NoiseAudio, int Steps, double Shift);

/// <summary>What a run publishes: the video blocks as RGB8, and the audio blocks as <c>[channel][sample]</c> float PCM, in block order.</summary>
public sealed record DuoOutput(IReadOnlyList<DuoDecodedBlock> Video, IReadOnlyList<float[]> Audio);

/// <summary>Runs one Duo conversation end to end: the chunk session denoises each block in order, and the clean latents stream through the video
/// and audio decoders. The video rows of a block are unpacked into the decoder's <c>[1, 24, 5, H, W]</c> chunk with the H3 video layout. The
/// decoders publish one block late, so the run ends with their End calls. The caller owns the decoders and the cache.</summary>
public sealed class DuoSession
{
    private const int ChunkLatents = 5;

    private readonly DuoChunkSession _chunks;
    private readonly DuoVideoStreamDecoder _video;
    private readonly DuoAudioStreamDecoder _audio;
    private readonly MiniMaxH3Config _config;
    private readonly DuoSampleInput _sample;
    private bool _ran;

    public DuoSession(MiniMaxH3Transformer transformer, IBackend backend, MiniMaxH3Config config, DuoSampleInput sample, DuoRopePolicy rope,
        DuoStreamCache cache, DuoVideoStreamDecoder video, DuoAudioStreamDecoder audio)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _sample = sample ?? throw new ArgumentNullException(nameof(sample));
        _video = video ?? throw new ArgumentNullException(nameof(video));
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _chunks = new DuoChunkSession(transformer ?? throw new ArgumentNullException(nameof(transformer)), backend, sample, rope,
            cache ?? throw new ArgumentNullException(nameof(cache)));
    }

    /// <summary>Denoises every block of the sample in order and returns what the decoders publish. Every block must be a duo_5_17 chunk of
    /// five latent steps, and the decoders must be built for the same blocks.</summary>
    /// <param name="onStep">Called after each denoise step with the number of steps finished over the whole run.</param>
    /// <param name="cancel">Checked before the prefill and before every denoise step.</param>
    public DuoOutput Run(DuoRun run, Action<int>? onStep = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(run);
        if (_ran)
        {
            throw new InvalidOperationException("A DuoSession runs once; make a new session for another conversation.");
        }
        int blocks = _sample.Blocks.Count;
        if (run.BodyText.Count != blocks || run.NoiseVideo.Count != blocks || run.NoiseAudio.Count != blocks)
        {
            throw new ArgumentException($"The run needs body text, video noise and audio noise for each of the {blocks} blocks.", nameof(run));
        }
        // Every block is checked before anything is committed: a grid or audio length that the decoders cannot take fails here, not mid-run.
        for (int block = 0; block < blocks; block++)
        {
            DuoBlockInput input = _sample.Blocks[block];
            if (input.Video.Frames != ChunkLatents)
            {
                throw new ArgumentException($"Block {block} has {input.Video.Frames} latent frames; a duo_5_17 chunk has {ChunkLatents}.", nameof(run));
            }
            if (input.AudioSteps != _audio.BlockSteps)
            {
                throw new ArgumentException($"Block {block} has {input.AudioSteps} audio steps; the audio stream takes {_audio.BlockSteps}.", nameof(run));
            }
        }
        cancel.ThrowIfCancellationRequested();
        _ran = true;
        _chunks.Prefill(run.SharedText, run.ReferenceVideo);

        List<DuoDecodedBlock> video = [];
        List<float[]> audio = [];
        for (int block = 0; block < blocks; block++)
        {
            int done = block * run.Steps;
            (float[] rows, float[] audioRows) = _chunks.GenerateBlock(block, run.BodyText[block], run.NoiseVideo[block], run.NoiseAudio[block],
                run.Steps, run.Shift, onStep is null ? null : step => onStep(done + step), cancel);
            using (Tensor chunk = Chunk(rows, block))
            {
                video.AddRange(_video.OnChunk(chunk));
            }
            Publish(audio, _audio.OnBlock(audioRows));
        }
        video.AddRange(_video.End());
        Publish(audio, _audio.End());
        return new DuoOutput(video, audio);
    }

    /// <summary>The video rows of one block as the decoder's chunk <c>[1, 24, 5, H, W]</c>, on the backend.</summary>
    private Tensor Chunk(float[] rows, int block)
    {
        DuoGrid grid = _sample.Blocks[block].Video;
        if (grid.Frames != ChunkLatents)
        {
            throw new InvalidOperationException($"Block {block} has {grid.Frames} latent frames; a duo_5_17 chunk has {ChunkLatents}.");
        }
        Tensor packed = DuoDeviceWindow.Upload(new TensorShape(grid.VideoRows, _config.VideoPatchDim), rows);
        try
        {
            return MiniMaxH3Latents.UnpackVideo(packed, grid.Frames, grid.Height, grid.Width, _config);
        }
        finally
        {
            packed.Dispose();
        }
    }

    private static void Publish(List<float[]> audio, float[] block)
    {
        if (block.Length > 0)
        {
            audio.Add(block);
        }
    }
}
