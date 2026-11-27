using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Streams Duo audio latent blocks through a windowed audio decoder. A block is <c>blockSteps</c> latent steps per channel, two
/// channels, <c>width</c> values per step, given as rows in the order [channel][step][width] and turned into the decoder's layout by
/// <see cref="MiniMaxH3Latents.UnpackAudio"/>. The decoder runs on a window of up to <c>contextSteps</c> earlier steps, the block, and the first
/// <c>lookaheadSteps</c> steps of the next block; only the block's samples are kept. A block is published when its successor arrives, so
/// <see cref="End"/> publishes the last one. The retained history and the blocks are backend tensors, and the windows and crops are built by
/// <see cref="DuoDeviceWindow"/>, the same owner the video stream decoder uses.</summary>
/// <remarks><para>The decoder convolves over time, so a block's samples depend on latent steps on both sides. With a left context at least
/// the decoder's left receptive field and a lookahead at least its right one, each block equals the same samples of one decode of the whole
/// sequence. The tests check that on a tiny decoder with random weights, and a lookahead of zero is shown to change the output. The
/// receptive fields are not derived here, and the shipped context and lookahead are not yet chosen.</para>
/// <para>Each publish decodes <c>contextSteps + blockSteps + lookaheadSteps</c> steps to keep <c>blockSteps</c>. The cost relative to a decode
/// of the block alone is therefore <c>(contextSteps + blockSteps + lookaheadSteps) / blockSteps</c>: about 4.7x at 32 context, 12 block and
/// 12 lookahead steps.</para>
/// <para>Upstream's audio chunk policy is not checked against upstream: parity is not run.</para></remarks>
public sealed class DuoAudioStreamDecoder : IDisposable
{
    private const int Channels = 2;
    private const int StepAxis = 3;
    private const int SampleAxis = 2;

    private readonly IBackend _backend;
    private readonly Func<Tensor, Tensor> _decodeWindow;
    private readonly MiniMaxH3Config _rowLayout;
    private readonly int _width;
    private readonly int _samplesPerStep;
    private readonly int _blockSteps;
    private readonly int _lookaheadSteps;
    private readonly int _contextSteps;
    private Tensor? _pending;    // the block awaiting its lookahead, [1, width, 2, blockSteps]
    private Tensor? _history;    // the retained steps before the pending block, [1, width, 2, retained]
    private int _retained;
    private bool _ended;
    private bool _disposed;

    /// <param name="backend">Backend that owns the retained blocks and runs the window and crop copies.</param>
    /// <param name="decodeWindow">Decodes a window <c>[1, width, 2, steps]</c> into <c>[1, 2, steps · samplesPerStep]</c>. The window is owned
    /// by this class and disposed after the call; a returned tensor other than the window is disposed here.</param>
    /// <param name="width">Latent channels per step.</param>
    /// <param name="samplesPerStep">Samples the decoder produces per latent step.</param>
    /// <param name="blockSteps">Latent steps per channel in one block.</param>
    /// <param name="lookaheadSteps">Steps of the next block the window includes after the block, at most <paramref name="blockSteps"/>. Zero
    /// decodes the block with no right context.</param>
    /// <param name="contextSteps">Earlier steps the window keeps before a block.</param>
    public DuoAudioStreamDecoder(IBackend backend, Func<Tensor, Tensor> decodeWindow, int width, int samplesPerStep, int blockSteps,
        int lookaheadSteps, int contextSteps)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _decodeWindow = decodeWindow ?? throw new ArgumentNullException(nameof(decodeWindow));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(samplesPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSteps);
        ArgumentOutOfRangeException.ThrowIfNegative(lookaheadSteps);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(lookaheadSteps, blockSteps);
        ArgumentOutOfRangeException.ThrowIfNegative(contextSteps);
        _width = width;
        _samplesPerStep = samplesPerStep;
        _blockSteps = blockSteps;
        _lookaheadSteps = lookaheadSteps;
        _contextSteps = contextSteps;
        _rowLayout = new MiniMaxH3Config { AudioLatentsDim = width };
    }

    /// <summary>The H3 audio VAE decoder, run on one window at a time. Its sample rate and hop must match <paramref name="spec"/>, and each block
    /// is <see cref="DuoCodecSpec.AudioLatents"/> steps.</summary>
    /// <remarks>The stream does not preload the decoder's weights. <c>MiniMaxH3Pipeline</c> preloads the audio VAE before the decode and frees it
    /// after, so the caller keeps the decoder resident for the stream's life.</remarks>
    /// <exception cref="InvalidOperationException">The decoder's sample rate or samples per latent step differ from the duo_5_17 codec.</exception>
    public static DuoAudioStreamDecoder ForH3(MiniMaxH3AudioVaeDecoder decoder, IBackend backend, DuoCodecSpec spec, int lookaheadSteps,
        int contextSteps)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(spec);
        if (decoder.SampleRate != DuoCodecSpec.SampleRate || decoder.SamplesPerLatentFrame != DuoCodecSpec.AudioHop)
        {
            throw new InvalidOperationException(
                $"The audio decoder runs at {decoder.SampleRate} Hz with {decoder.SamplesPerLatentFrame} samples per step; the Duo codec needs {DuoCodecSpec.SampleRate} Hz and {DuoCodecSpec.AudioHop}.");
        }
        return new DuoAudioStreamDecoder(backend, window => decoder.Decode(backend, window), decoder.LatentChannels,
            decoder.SamplesPerLatentFrame, spec.AudioLatents, lookaheadSteps, contextSteps);
    }

    /// <summary>Latent steps per channel in one block; every block the stream receives has this many.</summary>
    public int BlockSteps => _blockSteps;

    /// <summary>Receives the next block, <c>2 · blockSteps · width</c> values in [channel][step][width] order. Returns the samples of the
    /// previous block, [channel][sample] with <c>blockSteps · samplesPerStep</c> per channel, or nothing for the first block.</summary>
    public float[] OnBlock(float[] rows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(rows);
        if (_ended)
        {
            throw new InvalidOperationException("The stream has ended; no block can follow End.");
        }
        int expected = Channels * _blockSteps * _width;
        if (rows.Length != expected)
        {
            throw new ArgumentException($"A block has {rows.Length} values; expected {Channels} channels of {_blockSteps} steps of {_width}.", nameof(rows));
        }
        Tensor block = Block(rows);
        if (_pending is null)
        {
            _pending = block;
            return [];
        }
        float[] published;
        try
        {
            published = Publish(_pending, block);
        }
        catch
        {
            block.Dispose();
            throw;
        }
        _pending.Dispose();
        _pending = block;
        return published;
    }

    /// <summary>Publishes the last block, which has no lookahead, and ends the stream. Returns nothing when no block was received.</summary>
    public float[] End()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_ended)
        {
            throw new InvalidOperationException("The stream has already ended.");
        }
        float[] published = _pending is null ? [] : Publish(_pending, next: null);
        _pending?.Dispose();
        _pending = null;
        _ended = true;
        return published;
    }

    /// <summary>Frees the pending block and the retained history.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _pending?.Dispose();
        _history?.Dispose();
        _pending = null;
        _history = null;
    }

    /// <summary>The rows of one block as the decoder's layout: the H3 audio unpack of [channel][step][width] rows.</summary>
    private Tensor Block(float[] rows)
    {
        Tensor packed = DuoDeviceWindow.Upload([Channels * _blockSteps, _width], rows);
        try
        {
            return MiniMaxH3Latents.UnpackAudio(packed, _blockSteps, _rowLayout);
        }
        finally
        {
            packed.Dispose();
        }
    }

    /// <summary>Decodes the pending block with the first <c>lookaheadSteps</c> steps of <paramref name="next"/> after it, and advances.</summary>
    private float[] Publish(Tensor block, Tensor? next)
    {
        int ahead = next is null ? 0 : _lookaheadSteps;
        int steps = _retained + _blockSteps + ahead;
        List<DuoDeviceWindow.Part> parts = [];
        if (_history is not null)
        {
            parts.Add(new DuoDeviceWindow.Part(_history, 0, _retained));
        }
        parts.Add(new DuoDeviceWindow.Part(block, 0, _blockSteps));
        if (next is not null)
        {
            parts.Add(new DuoDeviceWindow.Part(next, 0, _lookaheadSteps));
        }
        Tensor window = DuoDeviceWindow.Concat(_backend, StepAxis, parts);
        float[] samples;
        try
        {
            Tensor pixels = _decodeWindow(window);
            try
            {
                samples = Crop(pixels, steps);
            }
            finally
            {
                // A decoder may hand back the window itself; it is disposed once, below.
                if (!ReferenceEquals(pixels, window))
                {
                    pixels.Dispose();
                }
            }
        }
        finally
        {
            window.Dispose();
        }
        Advance(block);
        return samples;
    }

    /// <summary>Keeps the block's samples, skipping the retained history, from a decoded <c>[1, 2, steps · samplesPerStep]</c> window.</summary>
    private float[] Crop(Tensor audio, int steps)
    {
        int perStep = _samplesPerStep;
        if (audio.Shape.Rank != 3 || audio.Shape[0] != 1 || audio.Shape[1] != Channels || audio.Shape[2] != (long)steps * perStep)
        {
            throw new InvalidOperationException(
                $"The window decoder returned {audio.Shape} for {steps} steps; expected [1, {Channels}, {steps * perStep}].");
        }
        Tensor kept = DuoDeviceWindow.Slice(_backend, audio, SampleAxis, _retained * perStep, _blockSteps * perStep);
        try
        {
            return kept.AsSpan<float>().ToArray();
        }
        finally
        {
            kept.Dispose();
        }
    }

    /// <summary>Keeps the last <c>contextSteps</c> steps of the history followed by the block just published.</summary>
    private void Advance(Tensor block)
    {
        int total = _retained + _blockSteps;
        int keep = Math.Min(_contextSteps, total);
        if (keep == 0)
        {
            _history?.Dispose();
            _history = null;
            _retained = 0;
            return;
        }
        List<DuoDeviceWindow.Part> parts = [];
        if (_history is not null)
        {
            parts.Add(new DuoDeviceWindow.Part(_history, 0, _retained));
        }
        parts.Add(new DuoDeviceWindow.Part(block, 0, _blockSteps));
        Tensor merged = DuoDeviceWindow.Concat(_backend, StepAxis, parts);
        Tensor next;
        try
        {
            next = DuoDeviceWindow.Slice(_backend, merged, StepAxis, total - keep, keep);
        }
        finally
        {
            merged.Dispose();
        }
        _history?.Dispose();
        _history = next;
        _retained = keep;
    }
}
