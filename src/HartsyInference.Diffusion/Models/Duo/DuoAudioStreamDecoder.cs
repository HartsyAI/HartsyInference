using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Streams Duo audio latent blocks through a windowed audio decoder. A block is <c>blockSteps</c> latent steps per channel, two
/// channels, <c>width</c> values per step, given as rows in the order [channel][step][width]. The decoder is run on a window of up to
/// <c>contextSteps</c> earlier steps, the block, and the first <c>lookaheadSteps</c> steps of the next block, and only the block's samples
/// are kept, as [channel][sample]. A block is published when its successor arrives, so <see cref="End"/> publishes the last one.</summary>
/// <remarks><para>The decoder convolves over time, so a block's samples depend on latent steps on both sides. With a left context at least
/// the decoder's left receptive field and a lookahead at least its right one, each block equals the same samples of one decode of the whole
/// sequence. The tests check that on a tiny decoder with random weights, and a lookahead of zero is shown to change the output. The
/// receptive fields are not derived here, the shipped context and lookahead are not yet chosen, and upstream's audio chunk policy and the DiT
/// audio row order are not checked against upstream: parity is not run.</para>
/// <para>Each publish decodes <c>contextSteps + blockSteps + lookaheadSteps</c> steps to keep <c>blockSteps</c>. The cost relative to a decode
/// of the block alone is therefore <c>(contextSteps + blockSteps + lookaheadSteps) / blockSteps</c>: about 4.7x at 32 context, 12 block and
/// 12 lookahead steps.</para></remarks>
public sealed class DuoAudioStreamDecoder
{
    private const int Channels = 2;

    private readonly Func<Tensor, Tensor> _decodeWindow;
    private readonly int _width;
    private readonly int _samplesPerStep;
    private readonly int _blockSteps;
    private readonly int _lookaheadSteps;
    private readonly int _contextSteps;
    private float[]? _pending;
    private float[][] _history;   // per channel, the last retained steps as [step][width]
    private int _retained;
    private bool _ended;

    /// <param name="decodeWindow">Decodes a window <c>[1, width, 2, steps]</c> into <c>[1, 2, steps · samplesPerStep]</c>. The window is
    /// owned by this class and disposed after the call. A returned tensor other than the window is disposed here; the window itself is never
    /// disposed twice.</param>
    /// <param name="width">Latent channels per step.</param>
    /// <param name="samplesPerStep">Samples the decoder produces per latent step.</param>
    /// <param name="blockSteps">Latent steps per channel in one block.</param>
    /// <param name="lookaheadSteps">Steps of the next block the window includes after the block, at most <paramref name="blockSteps"/>. Zero
    /// decodes the block with no right context.</param>
    /// <param name="contextSteps">Earlier steps the window keeps before a block.</param>
    public DuoAudioStreamDecoder(Func<Tensor, Tensor> decodeWindow, int width, int samplesPerStep, int blockSteps, int lookaheadSteps, int contextSteps)
    {
        ArgumentNullException.ThrowIfNull(decodeWindow);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(samplesPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSteps);
        ArgumentOutOfRangeException.ThrowIfNegative(lookaheadSteps);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(lookaheadSteps, blockSteps);
        ArgumentOutOfRangeException.ThrowIfNegative(contextSteps);
        _decodeWindow = decodeWindow;
        _width = width;
        _samplesPerStep = samplesPerStep;
        _blockSteps = blockSteps;
        _lookaheadSteps = lookaheadSteps;
        _contextSteps = contextSteps;
        _history = [Array.Empty<float>(), Array.Empty<float>()];
    }

    /// <summary>The H3 audio VAE decoder, run on one window at a time.</summary>
    public static DuoAudioStreamDecoder ForH3(MiniMaxH3AudioVaeDecoder decoder, IBackend backend, int blockSteps, int lookaheadSteps, int contextSteps)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(backend);
        return new DuoAudioStreamDecoder(window => decoder.Decode(backend, window), decoder.LatentChannels, decoder.SamplesPerLatentFrame,
            blockSteps, lookaheadSteps, contextSteps);
    }

    /// <summary>Receives the next block, <c>2 · blockSteps · width</c> values in [channel][step][width] order. Returns the samples of the
    /// previous block, [channel][sample] with <c>blockSteps · samplesPerStep</c> per channel, or nothing for the first block.</summary>
    public float[] OnBlock(float[] rows)
    {
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
        if (_pending is null)
        {
            _pending = (float[])rows.Clone();
            return [];
        }
        float[] published = Publish(_pending, rows);
        _pending = (float[])rows.Clone();
        return published;
    }

    /// <summary>Publishes the last block, which has no lookahead, and ends the stream. Returns nothing when no block was received.</summary>
    public float[] End()
    {
        if (_ended)
        {
            throw new InvalidOperationException("The stream has already ended.");
        }
        float[] published = _pending is null ? [] : Publish(_pending, next: null);
        _pending = null;
        _ended = true;
        return published;
    }

    /// <summary>Decodes the pending block with the first <c>lookaheadSteps</c> steps of <paramref name="next"/> after it, and advances.</summary>
    private float[] Publish(float[] block, float[]? next)
    {
        int ahead = next is null ? 0 : _lookaheadSteps;
        int steps = _retained + _blockSteps + ahead;
        float[] samples;
        Tensor window = Window(block, next, steps);
        try
        {
            Tensor audio = _decodeWindow(window);
            try
            {
                samples = Crop(audio, steps);
            }
            finally
            {
                if (!ReferenceEquals(audio, window))
                {
                    audio.Dispose();
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

    /// <summary>The window in <c>[1, width, 2, steps]</c> order: retained history, the block, then the lookahead steps.</summary>
    private Tensor Window(float[] block, float[]? next, int steps)
    {
        Tensor window = new(new TensorShape(1, _width, Channels, steps), DType.F32);
        Span<float> w = window.AsSpan<float>();
        for (int c = 0; c < Channels; c++)
        {
            for (int d = 0; d < _width; d++)
            {
                for (int t = 0; t < steps; t++)
                {
                    float v;
                    if (t < _retained)
                    {
                        v = _history[c][t * _width + d];
                    }
                    else if (t < _retained + _blockSteps)
                    {
                        v = block[(c * _blockSteps + (t - _retained)) * _width + d];
                    }
                    else
                    {
                        v = next![(c * _blockSteps + (t - _retained - _blockSteps)) * _width + d];
                    }
                    w[(d * Channels + c) * steps + t] = v;
                }
            }
        }
        return window;
    }

    /// <summary>Keeps the samples of the block, skipping the retained history, from a decoded <c>[1, 2, steps · samplesPerStep]</c> window.</summary>
    private float[] Crop(Tensor audio, int steps)
    {
        int perStep = _samplesPerStep;
        if (audio.Shape.Rank != 3 || audio.Shape[0] != 1 || audio.Shape[1] != Channels || audio.Shape[2] != (long)steps * perStep)
        {
            throw new InvalidOperationException(
                $"The window decoder returned {audio.Shape} for {steps} steps; expected [1, {Channels}, {steps * perStep}].");
        }
        int windowSamples = steps * perStep, skip = _retained * perStep, keep = _blockSteps * perStep;
        ReadOnlySpan<float> a = audio.AsSpan<float>();
        float[] samples = new float[Channels * keep];
        for (int c = 0; c < Channels; c++)
        {
            a.Slice(c * windowSamples + skip, keep).CopyTo(samples.AsSpan(c * keep, keep));
        }
        return samples;
    }

    /// <summary>Keeps the last <c>contextSteps</c> steps of the history followed by the block just published.</summary>
    private void Advance(float[] block)
    {
        int total = _retained + _blockSteps;
        int keep = Math.Min(_contextSteps, total);
        float[][] nextHistory = new float[Channels][];
        for (int c = 0; c < Channels; c++)
        {
            nextHistory[c] = new float[keep * _width];
            for (int t = 0; t < keep; t++)
            {
                int source = total - keep + t;
                for (int d = 0; d < _width; d++)
                {
                    nextHistory[c][t * _width + d] = source < _retained
                        ? _history[c][source * _width + d]
                        : block[(c * _blockSteps + (source - _retained)) * _width + d];
                }
            }
        }
        _history = nextHistory;
        _retained = keep;
    }
}
