using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Heads;
using Xunit;
using static HartsyInference.Diffusion.Tests.TinyWanCheckpoint;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The convolutional pixel head and the output-head seam on the Wan transformer, on the CPU backend. The head is
/// checked against a plain-loop implementation of the layer sequence written from the paper's description (the same author,
/// so it catches layout and indexing mistakes, not a misreading); the paper's authors' code is not released. The parameter
/// counts check that the layer shapes this head takes reproduce the 44M, 337M and 942M the paper states.</summary>
public sealed class ConvPixelHeadTests
{
    private const int HeadWidth = 6;
    private const int FirstWidth = 5;
    private const int SecondWidth = 4;
    private const int GridT = 3, GridH = 2, GridW = 3;
    private const int OutChannels = 3;

    [Fact]
    public void HeadMatchesAPlainLoopImplementationOfTheSameLayers()
    {
        using CpuBackend backend = new();
        using Weights weights = new(HeadWidth, [FirstWidth, SecondWidth], withBias: true, seed: 5);
        ConvPixelHead head = new(HeadWidth, weights.Build());
        int tokens = GridT * GridH * GridW;
        using Tensor hidden = IrisZTestData.Random(new Random(6), tokens, HeadWidth);
        using Tensor embedding = IrisZTestData.Random(new Random(7), 1, HeadWidth);

        using Tensor actual = head.Forward(backend, hidden, embedding, (GridT, GridH, GridW));
        float[] expected = Reference(weights, hidden.AsReadOnlySpan<float>().ToArray(), embedding.AsReadOnlySpan<float>().ToArray());

        Assert.Equal(new TensorShape([1L, OutChannels, GridT, GridH * 4, GridW * 4]), actual.Shape);
        float worst = 0f;
        ReadOnlySpan<float> values = actual.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(values[i] - expected[i]));
        Assert.True(worst <= 1e-4f, $"max abs difference {worst} from the plain-loop reference");
    }

    [Fact]
    public void HeadIsCausalInTime()
    {
        using CpuBackend backend = new();
        using Weights weights = new(HeadWidth, [FirstWidth, SecondWidth], withBias: true, seed: 5);
        ConvPixelHead head = new(HeadWidth, weights.Build());
        int tokensPerFrame = GridH * GridW;
        using Tensor hidden = IrisZTestData.Random(new Random(6), GridT * tokensPerFrame, HeadWidth);
        using Tensor embedding = IrisZTestData.Random(new Random(7), 1, HeadWidth);
        using Tensor changed = new(hidden.Shape, DType.F32);
        hidden.AsReadOnlySpan<float>().CopyTo(changed.AsSpan<float>());
        // Change only the last frame's tokens: earlier output frames must not move.
        Span<float> last = changed.AsSpan<float>()[((GridT - 1) * tokensPerFrame * HeadWidth)..];
        for (int i = 0; i < last.Length; i++) last[i] += 0.5f;

        using Tensor before = head.Forward(backend, hidden, embedding, (GridT, GridH, GridW));
        using Tensor after = head.Forward(backend, changed, embedding, (GridT, GridH, GridW));

        int frameValues = GridH * 4 * GridW * 4;
        for (int channel = 0; channel < OutChannels; channel++)
        {
            for (int frame = 0; frame < GridT; frame++)
            {
                ReadOnlySpan<float> a = before.AsReadOnlySpan<float>().Slice((channel * GridT + frame) * frameValues, frameValues);
                ReadOnlySpan<float> b = after.AsReadOnlySpan<float>().Slice((channel * GridT + frame) * frameValues, frameValues);
                if (frame < GridT - 1) Assert.True(a.SequenceEqual(b), $"frame {frame} of channel {channel} changed when only the last frame's tokens did");
                else Assert.False(a.SequenceEqual(b), "the last frame did not respond to its own tokens");
            }
        }
    }

    [Fact]
    public void WeightShapesAreValidated()
    {
        using Weights weights = new(HeadWidth, [FirstWidth, SecondWidth], withBias: false, seed: 5);

        Assert.Throws<ArgumentException>(() => new ConvPixelHead(HeadWidth + 1, weights.Build()));
        Assert.Throws<ArgumentException>(() => new ConvPixelHead(HeadWidth, weights.Build() with { UpBlocks = [] }));
        ConvPixelHeadWeights good = weights.Build();
        using Tensor wrongOutput = IrisZTestData.Random(new Random(1), OutChannels, SecondWidth + 1, 3, 1, 1);
        Assert.Throws<ArgumentException>(() => new ConvPixelHead(HeadWidth, good with { OutputWeight = wrongOutput }));
    }

    [Fact]
    public void ForwardRejectsAWrongHiddenShapeAndAPerFrameEmbedding()
    {
        using CpuBackend backend = new();
        using Weights weights = new(HeadWidth, [FirstWidth], withBias: false, seed: 5);
        ConvPixelHead head = new(HeadWidth, weights.Build());
        using Tensor hidden = IrisZTestData.Random(new Random(1), GridT * GridH * GridW, HeadWidth);
        using Tensor tooFew = IrisZTestData.Random(new Random(1), 5, HeadWidth);
        using Tensor perFrame = IrisZTestData.Random(new Random(1), GridT, HeadWidth);
        using Tensor embedding = IrisZTestData.Random(new Random(1), 1, HeadWidth);

        Assert.Throws<ArgumentException>(() => head.Forward(backend, tooFew, embedding, (GridT, GridH, GridW)));
        Assert.Throws<ArgumentException>(() => head.Forward(backend, hidden, perFrame, (GridT, GridH, GridW)));
    }

    /// <summary>The paper states 44M, 337M and 942M head parameters for its three configurations. Counting the layers this
    /// head takes (a 3x1x1 bottleneck D to D, per up block a 1x2x2 transposed convolution C to C and a 3x3x3 convolution to the
    /// block's width, a 3x1x1 output convolution), weights only, reproduces all three.</summary>
    [Theory]
    [InlineData(1536, new[] { 512, 256, 128, 64 }, 44)]
    [InlineData(5120, new[] { 1024, 256, 64, 64 }, 337)]
    [InlineData(5120, new[] { 4096, 1024, 256, 64, 64 }, 942)]
    public void LayerShapesReproduceTheParameterCountsThePaperStates(int dim, int[] widths, int millions)
    {
        long count = 3L * dim * dim;
        int channels = dim;
        foreach (int width in widths)
        {
            count += 4L * channels * channels + 27L * channels * width;
            channels = width;
        }
        count += 3L * channels * OutChannels;

        Assert.Equal(millions, (int)Math.Round(count / 1e6));
    }

    [Fact]
    public void TransformerWithAnOutputHeadNeedsNoProjectionAndReturnsTheHeadsOutput()
    {
        using CpuBackend backend = new();
        // A pixel DiT: 3-channel frames, 1x4x4 patches, no proj_out / scale_shift_table in the checkpoint.
        using TinyWanCheckpoint checkpoint = new(inChannels: 3, patchHeight: 4, patchWidth: 4, withProjection: false);
        WanVideoConfig config = new()
        {
            NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn, InChannels = 3, OutChannels = 3,
            VaeLatentChannels = 3, TextDim = TextDim, FreqDim = FreqDim, PatchSize = (1, 4, 4),
        };
        using Weights headWeights = new(Inner, [8, 6], withBias: true, seed: 9);
        ConvPixelHead head = new(Inner, headWeights.Build());
        using WanVideoTransformer transformer = new(config) { OutputHead = head };
        transformer.LoadWeights(checkpoint.Weights);
        List<Tensor> shared = transformer.EnumerateSharedWeights().ToList();
        Assert.All(head.EnumerateWeights(), weight => Assert.Contains(weight, shared));
        using Tensor frames = IrisZTestData.Random(new Random(11), 1, 3, 3, 8, 8);
        using Tensor text = IrisZTestData.Random(new Random(12), 5, TextDim);

        using Tensor output = transformer.Forward(backend, frames, text, 0.6f);

        Assert.Equal(new TensorShape([1L, 3, 3, 8, 8]), output.Shape);
        foreach (float value in output.AsReadOnlySpan<float>()) Assert.True(float.IsFinite(value));
    }

    [Fact]
    public void WithoutAnOutputHeadAMissingProjectionStillFailsAsBefore()
    {
        using TinyWanCheckpoint checkpoint = new(withProjection: false);
        using WanVideoTransformer transformer = new(new WanVideoConfig
        {
            NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn, InChannels = Channels, OutChannels = Channels,
            VaeLatentChannels = Channels, TextDim = TextDim, FreqDim = FreqDim,
        });

        Assert.Throws<KeyNotFoundException>(() => transformer.LoadWeights(checkpoint.Weights));
    }

    [Fact]
    public void AHeadWhoseUpsamplingDoesNotMatchThePatchSizeIsRefused()
    {
        using Weights oneBlock = new(Inner, [8], withBias: false, seed: 9);
        ConvPixelHead head = new(Inner, oneBlock.Build());
        WanVideoConfig patch4 = new()
        {
            NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn, InChannels = 3, OutChannels = 3,
            VaeLatentChannels = 3, TextDim = TextDim, FreqDim = FreqDim, PatchSize = (1, 4, 4),
        };

        Assert.Equal((1, 2, 2), head.Upsample);
        Assert.Throws<ArgumentException>(() => new WanVideoTransformer(patch4) { OutputHead = head });
    }

    [Fact]
    public void HeadRefusesATokenOrEmbeddingThatIsNotF32()
    {
        using CpuBackend backend = new();
        using Weights weights = new(HeadWidth, [FirstWidth], withBias: false, seed: 5);
        ConvPixelHead head = new(HeadWidth, weights.Build());
        using Tensor hidden = IrisZTestData.Random(new Random(1), GridT * GridH * GridW, HeadWidth);
        using Tensor embedding = IrisZTestData.Random(new Random(2), 1, HeadWidth);
        using Tensor halfHidden = hidden.CastTo(DType.F16);

        Assert.Throws<ArgumentException>(() => head.Forward(backend, halfHidden, embedding, (GridT, GridH, GridW)));
    }

    [Fact]
    public void AnOutputHeadRefusesPerFrameTimesteps()
    {
        using CpuBackend backend = new();
        using TinyWanCheckpoint checkpoint = new(inChannels: 3, patchHeight: 4, patchWidth: 4, withProjection: false);
        WanVideoConfig config = new()
        {
            NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn, InChannels = 3, OutChannels = 3,
            VaeLatentChannels = 3, TextDim = TextDim, FreqDim = FreqDim, PatchSize = (1, 4, 4),
        };
        using Weights headWeights = new(Inner, [8, 6], withBias: false, seed: 9);
        using WanVideoTransformer transformer = new(config) { OutputHead = new ConvPixelHead(Inner, headWeights.Build()) };
        transformer.LoadWeights(checkpoint.Weights);
        using Tensor frames = IrisZTestData.Random(new Random(11), 1, 3, 3, 4, 4);
        using Tensor text = IrisZTestData.Random(new Random(12), 5, TextDim);

        Assert.Throws<NotSupportedException>(() => transformer.Forward(backend, frames, text, [0f, 0.5f, 0.5f]));
    }

    // ── plain-loop reference ──

    /// <summary>x[c][t][y][x] flattened as ((c * T + t) * H + y) * W + x.</summary>
    private static float[] Reference(Weights weights, float[] tokens, float[] embedding)
    {
        int t = GridT, h = GridH, w = GridW, d = weights.InputDim;
        float[] x = new float[d * t * h * w];
        for (int s = 0; s < t * h * w; s++)
            for (int c = 0; c < d; c++) x[c * t * h * w + s] = Silu(tokens[s * d + c] + embedding[c]);
        x = CausalConv(x, d, t, h, w, weights.BottleneckWeight, weights.BottleneckBias, d, 3, 1, 1);
        int channels = d;
        for (int i = 0; i < weights.Widths.Length; i++)
        {
            ConvPixelUpBlockWeights block = weights.Blocks[i];
            int mid = (int)block.TransposeWeight.Shape[1];
            x = TransposeConv(x, channels, mid, t, h, w, block.TransposeWeight.AsReadOnlySpan<float>().ToArray(),
                block.TransposeBias?.AsReadOnlySpan<float>().ToArray());
            h *= 2;
            w *= 2;
            x = CausalConv(x, mid, t, h, w, block.ConvWeight, block.ConvBias, weights.Widths[i], 3, 3, 3);
            for (int j = 0; j < x.Length; j++) x[j] = Silu(x[j]);
            channels = weights.Widths[i];
        }
        return CausalConv(x, channels, t, h, w, weights.OutputWeight, weights.OutputBias, OutChannels, 3, 1, 1);
    }

    private static float Silu(float v) => v / (1f + MathF.Exp(-v));

    private static float[] TransposeConv(float[] x, int cIn, int cOut, int t, int h, int w, float[] weight, float[]? bias)
    {
        float[] output = new float[cOut * t * 2 * h * 2 * w];
        for (int co = 0; co < cOut; co++)
            for (int f = 0; f < t; f++)
                for (int y = 0; y < 2 * h; y++)
                    for (int xx = 0; xx < 2 * w; xx++)
                    {
                        float acc = bias?[co] ?? 0f;
                        for (int ci = 0; ci < cIn; ci++)
                            acc += x[((ci * t + f) * h + y / 2) * w + xx / 2] * weight[((ci * cOut + co) * 2 + y % 2) * 2 + xx % 2];
                        output[((co * t + f) * 2 * h + y) * 2 * w + xx] = acc;
                    }
        return output;
    }

    private static float[] CausalConv(float[] x, int cIn, int t, int h, int w, Tensor weight, Tensor? bias, int cOut, int kt, int kh, int kw)
    {
        float[] wv = weight.AsReadOnlySpan<float>().ToArray();
        float[]? bv = bias?.AsReadOnlySpan<float>().ToArray();
        float[] output = new float[cOut * t * h * w];
        for (int co = 0; co < cOut; co++)
            for (int f = 0; f < t; f++)
                for (int y = 0; y < h; y++)
                    for (int xx = 0; xx < w; xx++)
                    {
                        float acc = bv?[co] ?? 0f;
                        for (int ci = 0; ci < cIn; ci++)
                            for (int a = 0; a < kt; a++)
                            {
                                int ft = f - (kt - 1) + a;
                                if (ft < 0) continue;
                                for (int b = 0; b < kh; b++)
                                {
                                    int fy = y - (kh - 1) / 2 + b;
                                    if (fy < 0 || fy >= h) continue;
                                    for (int c = 0; c < kw; c++)
                                    {
                                        int fx = xx - (kw - 1) / 2 + c;
                                        if (fx < 0 || fx >= w) continue;
                                        acc += x[((ci * t + ft) * h + fy) * w + fx] * wv[(((co * cIn + ci) * kt + a) * kh + b) * kw + c];
                                    }
                                }
                            }
                        output[((co * t + f) * h + y) * w + xx] = acc;
                    }
        return output;
    }

    /// <summary>Random head weights in the shapes the head takes, owned by the test.</summary>
    private sealed class Weights : IDisposable
    {
        private readonly List<Tensor> _all = [];

        public Weights(int inputDim, int[] widths, bool withBias, int seed)
        {
            InputDim = inputDim;
            Widths = widths;
            Random rng = new(seed);
            BottleneckWeight = Make(rng, inputDim, inputDim, 3, 1, 1);
            BottleneckBias = withBias ? Make(rng, inputDim) : null;
            int channels = inputDim;
            foreach (int width in widths)
            {
                Blocks.Add(new ConvPixelUpBlockWeights(Make(rng, channels, channels, 2, 2), withBias ? Make(rng, channels) : null,
                    Make(rng, width, channels, 3, 3, 3), withBias ? Make(rng, width) : null));
                channels = width;
            }
            OutputWeight = Make(rng, OutChannels, channels, 3, 1, 1);
            OutputBias = withBias ? Make(rng, OutChannels) : null;
        }

        public int InputDim { get; }
        public int[] Widths { get; }
        public Tensor BottleneckWeight { get; }
        public Tensor? BottleneckBias { get; }
        public List<ConvPixelUpBlockWeights> Blocks { get; } = [];
        public Tensor OutputWeight { get; }
        public Tensor? OutputBias { get; }

        public ConvPixelHeadWeights Build() => new(BottleneckWeight, BottleneckBias, Blocks, OutputWeight, OutputBias);

        public void Dispose()
        {
            foreach (Tensor tensor in _all) tensor.Dispose();
        }

        private Tensor Make(Random rng, params long[] dims)
        {
            Tensor tensor = IrisZTestData.Random(rng, dims);
            _all.Add(tensor);
            return tensor;
        }
    }
}
