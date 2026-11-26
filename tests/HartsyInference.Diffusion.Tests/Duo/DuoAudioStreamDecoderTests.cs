using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo audio stream decoder on a tiny MiniMax-H3 audio VAE decoder with random weights. With a left context and a lookahead
/// at least the decoder's receptive field, the blocks concatenate to one full decode of the sequence. Dropping either the left context or
/// the lookahead changes the output. The shipped geometry is checked on the real decoder class, without weights. These are self-consistency
/// checks: parity against upstream is not run.</summary>
public sealed class DuoAudioStreamDecoderTests
{
    private const int Width = 32;            // LatentChannels, the shipped default
    private const int Blocks = 10;
    private const int SamplesPerStep = 4;   // the tiny decoder's upsample rates 2 × 2
    private const int Context = 32;
    /// <summary>Scale of the random conv weights. Large enough that the output peaks well above zero, so relative checks are meaningful.</summary>
    private const float WeightScale = 0.4f;
    /// <summary>Exactness is judged against the output's own peak, not an absolute bound that a small signal can pass trivially.</summary>
    private const float RelativeTolerance = 1e-4f;
    /// <summary>A dropped context or lookahead must move the output by at least this fraction of its peak, or the check would pass it.</summary>
    private const float Sensitivity = 1e-2f;

    private static readonly DuoCodecSpec Spec = DuoCodecSpec.Get("duo_5_17");
    private static readonly int BlockSteps = Spec.AudioLatents;
    private static readonly int Total = Blocks * BlockSteps;
    private static readonly int Lookahead = BlockSteps;

    [Fact]
    public void BlocksWithFullContextConcatenateToOneFullDecode()
    {
        using Tiny tiny = Tiny.Create();
        float[] full = Values(7, 2 * Total * Width, 0.5f);
        float[] expected = FullDecode(tiny, full);
        float peak = Peak(expected);
        Assert.True(peak > 0.05f, $"The reference output peaks at {peak}; the comparison below would be too weak to mean anything.");

        float[] actual = Stream(tiny, full, Lookahead, Context);

        Assert.Equal(expected.Length, actual.Length);
        float worst = MaxDiff(expected, actual);
        Assert.True(worst <= RelativeTolerance * peak, $"Largest difference {worst} exceeds {RelativeTolerance} of the peak {peak}.");
    }

    [Fact]
    public void WithoutLeftContextTheBlocksDisagreeWithTheFullDecode()
    {
        using Tiny tiny = Tiny.Create();
        float[] full = Values(7, 2 * Total * Width, 0.5f);
        float[] expected = FullDecode(tiny, full);
        float peak = Peak(expected);

        float[] actual = Stream(tiny, full, Lookahead, contextSteps: 0);

        Assert.True(MaxDiff(expected, actual) > Sensitivity * peak, "Expected the left context to matter.");
    }

    [Fact]
    public void WithoutLookaheadTheBlocksDisagreeWithTheFullDecode()
    {
        using Tiny tiny = Tiny.Create();
        float[] full = Values(7, 2 * Total * Width, 0.5f);
        float[] expected = FullDecode(tiny, full);
        float peak = Peak(expected);

        float[] actual = Stream(tiny, full, lookaheadSteps: 0, Context);

        Assert.True(MaxDiff(expected, actual) > Sensitivity * peak, "Expected the lookahead to matter.");
    }

    [Fact]
    public void ANoBlockStreamPublishesNothing()
    {
        using Tiny tiny = Tiny.Create();
        DuoAudioStreamDecoder stream = Make(tiny, Lookahead, Context);

        Assert.Empty(stream.End());
    }

    [Fact]
    public void ABlockOfTheWrongLengthIsRefused()
    {
        using Tiny tiny = Tiny.Create();
        DuoAudioStreamDecoder stream = Make(tiny, Lookahead, Context);

        Assert.Throws<ArgumentException>(() => stream.OnBlock(new float[2 * BlockSteps * Width - 1]));
    }

    [Fact]
    public void ALookaheadLongerThanABlockIsRefused()
    {
        using Tiny tiny = Tiny.Create();

        Assert.Throws<ArgumentOutOfRangeException>(() => Make(tiny, BlockSteps + 1, Context));
    }

    [Fact]
    public void NothingCanFollowEnd()
    {
        using Tiny tiny = Tiny.Create();
        DuoAudioStreamDecoder stream = Make(tiny, Lookahead, Context);
        stream.OnBlock(new float[2 * BlockSteps * Width]);
        stream.End();

        Assert.Throws<InvalidOperationException>(() => stream.OnBlock(new float[2 * BlockSteps * Width]));
        Assert.Throws<InvalidOperationException>(() => stream.End());
    }

    [Fact]
    public void ADecodeThatReturnsItsWindowIsRefusedWithoutDisposingItTwice()
    {
        // The decoder hands back the window it was given. That is an invalid shape for the crop, so publishing throws; the window is disposed
        // once on the way out. The throw is the observable outcome; the dispose count is not visible to the test.
        int calls = 0;
        DuoAudioStreamDecoder stream = new(TestBackends.Cpu, window =>
        {
            calls++;
            return window;
        }, Width, SamplesPerStep, BlockSteps, Lookahead, Context);
        stream.OnBlock(new float[2 * BlockSteps * Width]);

        Assert.Throws<InvalidOperationException>(() => stream.OnBlock(new float[2 * BlockSteps * Width]));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void ForH3AcceptsTheShippedGeometryAndRefusesAnotherSampleRate()
    {
        // The geometry check runs before any decode, so neither decoder needs weights.
        using MiniMaxH3AudioVaeDecoder shipped = new();
        Assert.NotNull(DuoAudioStreamDecoder.ForH3(shipped, TestBackends.Cpu, Spec, lookaheadSteps: 0, contextSteps: 0));

        using MiniMaxH3AudioVaeDecoder other = new(new MiniMaxH3AudioVaeConfig { SampleRate = 16000 });
        Assert.Throws<InvalidOperationException>(() => DuoAudioStreamDecoder.ForH3(other, TestBackends.Cpu, Spec, 0, 0));
    }

    private static DuoAudioStreamDecoder Make(Tiny tiny, int lookaheadSteps, int contextSteps) =>
        new(TestBackends.Cpu, window => tiny.Decoder.Decode(TestBackends.Cpu, window), Width, SamplesPerStep, BlockSteps, lookaheadSteps,
            contextSteps);

    /// <summary>Feeds every block to the stream and gathers what it publishes, [channel][sample] over the whole sequence.</summary>
    private static float[] Stream(Tiny tiny, float[] full, int lookaheadSteps, int contextSteps)
    {
        DuoAudioStreamDecoder stream = Make(tiny, lookaheadSteps, contextSteps);
        int keep = BlockSteps * SamplesPerStep;
        float[] gathered = new float[2 * Blocks * keep];
        int published = 0;
        void Place(float[] piece)
        {
            if (piece.Length == 0)
            {
                return;
            }
            Assert.Equal(2 * keep, piece.Length);
            for (int c = 0; c < 2; c++)
            {
                piece.AsSpan(c * keep, keep).CopyTo(gathered.AsSpan(c * Blocks * keep + published * keep, keep));
            }
            published++;
        }
        for (int b = 0; b < Blocks; b++)
        {
            Place(stream.OnBlock(BlockRows(full, b)));
        }
        Place(stream.End());
        Assert.Equal(Blocks, published);
        return gathered;
    }

    /// <summary>One decode of the whole latent sequence, [channel][sample].</summary>
    private static float[] FullDecode(Tiny tiny, float[] full)
    {
        float[] window = new float[Width * 2 * Total];
        for (int c = 0; c < 2; c++)
        {
            for (int t = 0; t < Total; t++)
            {
                for (int d = 0; d < Width; d++)
                {
                    window[(d * 2 + c) * Total + t] = full[(c * Total + t) * Width + d];
                }
            }
        }
        using Tensor latent = Make(window, 1, Width, 2, Total);
        using Tensor decoded = tiny.Decoder.Decode(TestBackends.Cpu, latent);
        return decoded.AsSpan<float>().ToArray();
    }

    /// <summary>Rows of block <paramref name="b"/> in [channel][step][width] order, cut from the full sequence.</summary>
    private static float[] BlockRows(float[] full, int b)
    {
        float[] rows = new float[2 * BlockSteps * Width];
        for (int c = 0; c < 2; c++)
        {
            for (int t = 0; t < BlockSteps; t++)
            {
                for (int d = 0; d < Width; d++)
                {
                    rows[(c * BlockSteps + t) * Width + d] = full[(c * Total + b * BlockSteps + t) * Width + d];
                }
            }
        }
        return rows;
    }

    private static float Peak(float[] values)
    {
        float peak = 0f;
        foreach (float v in values)
        {
            peak = MathF.Max(peak, MathF.Abs(v));
        }
        return peak;
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float worst = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        }
        return worst;
    }

    private static float[] Values(int seed, int count, float scale)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++)
        {
            v[i] = scale * MathF.Sin(seed * 1.3f + i * 0.61f);
        }
        return v;
    }

    private static Tensor Make(float[] values, params int[] dims)
    {
        Tensor t = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>A miniature H3 audio decoder with random weights, and the tensors it borrows. Only this test class uses it.</summary>
    private sealed class Tiny : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights;

        private Tiny(MiniMaxH3AudioVaeDecoder decoder, Dictionary<string, Tensor> weights)
        {
            Decoder = decoder;
            _weights = weights;
        }

        public MiniMaxH3AudioVaeDecoder Decoder { get; }

        public static Tiny Create()
        {
            Dictionary<string, Tensor> w = [];
            int seed = 1;
            void Add(string name, params int[] dims)
            {
                int count = dims.Aggregate(1, (a, d) => a * d);
                w[name] = Make(Values(seed++, count, WeightScale), dims);
            }
            Add("dec_in_proj.weight", 8, 32, 1);
            Add("dec_in_proj.bias", 8);
            Add("decoder.conv_pre.weight", 8, 8, 7);
            Add("decoder.conv_pre.bias", 8);
            Add("decoder.ups.0.0.weight", 8, 4, 4);
            Add("decoder.ups.0.0.bias", 4);
            Add("decoder.ups.1.0.weight", 4, 2, 4);
            Add("decoder.ups.1.0.bias", 2);
            foreach ((int n, int c) in new[] { (0, 4), (1, 2) })
            {
                Add($"decoder.resblocks.{n}.convs1.0.weight", c, c, 3);
                Add($"decoder.resblocks.{n}.convs1.0.bias", c);
                Add($"decoder.resblocks.{n}.convs2.0.weight", c, c, 3);
                Add($"decoder.resblocks.{n}.convs2.0.bias", c);
                for (int a = 0; a < 2; a++)
                {
                    Add($"decoder.resblocks.{n}.activations.{a}.act.alpha", c);
                    Add($"decoder.resblocks.{n}.activations.{a}.act.beta", c);
                }
            }
            Add("decoder.activation_post.act.alpha", 2);
            Add("decoder.activation_post.act.beta", 2);
            Add("decoder.conv_post.weight", 1, 2, 7);

            MiniMaxH3AudioVaeDecoder decoder = new(new MiniMaxH3AudioVaeConfig
            {
                DecoderInputChannels = 8,
                DecoderDim = 8,
                UpsampleRates = [2, 2],
                UpsampleKernels = [4, 4],
                ResblockKernels = [3],
                ResblockDilations = [[1]],
            });
            decoder.LoadWeights(w);
            return new Tiny(decoder, w);
        }

        public void Dispose()
        {
            Decoder.Dispose();
            foreach (Tensor t in _weights.Values)
            {
                t.Dispose();
            }
        }
    }
}
