using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat DiT on a tiny random checkpoint. The reference transcription runs in tests/python-reference/longcat_dit; these pin the
/// properties the reference's layout implies and that a Python run cannot see go wrong silently.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatTransformerTests : IDisposable
{
    private const int Frames = 9, Height = 4, Width = 4, TextRows = 7;
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;

    public LongCatTransformerTests() => _weights = LongCatTestData.Weights(_config, 1);

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    private float[] Velocity(IReadOnlyDictionary<string, Tensor> weights, Tensor latent, Tensor text, in LongCatConditioning conditioning,
        float timestep = 600f,
        int validText = -1, LongCatConfig? config = null)
    {
        using LongCatTransformer transformer = new(config ?? _config);
        transformer.LoadWeights(weights);
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(timestep, Frames), in conditioning,
            validText);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    [Fact]
    public void ForwardReturnsAFiniteVelocityOfTheLatentShape()
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        float[] velocity = Velocity(_weights, latent, text, LongCatConditioning.None);

        Assert.Equal(latent.ElementCount, velocity.Length);
        Assert.All(velocity, static v => Assert.True(float.IsFinite(v)));
        Assert.Contains(velocity, static v => MathF.Abs(v) > 1e-4f);
    }

    [Fact]
    public void ConditioningFramesDoNotDependOnTheNoisyLatents()
    {
        LongCatConditioning conditioning = new(NumCondLatents: 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using Tensor a = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor b = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        int perFrame = Height * Width;
        Span<float> bs = b.AsSpan<float>();
        for (int c = 0; c < _config.InChannels; c++)
            for (int i = 2 * perFrame; i < Frames * perFrame; i++)
                bs[c * Frames * perFrame + i] += 0.9f;

        float[] va = Velocity(_weights, a, text, in conditioning);
        float[] vb = Velocity(_weights, b, text, in conditioning);

        for (int c = 0; c < _config.OutChannels; c++)
        {
            ReadOnlySpan<float> condA = va.AsSpan(c * Frames * perFrame, 2 * perFrame), condB = vb.AsSpan(c * Frames * perFrame, 2 * perFrame);
            Assert.True(LongCatTestData.RelRms(condB, condA) < 1e-6, $"conditioning frames moved with the noisy latents (channel {c}).");
            ReadOnlySpan<float> noisyA = va.AsSpan(c * Frames * perFrame + 2 * perFrame, 7 * perFrame);
            ReadOnlySpan<float> noisyB = vb.AsSpan(c * Frames * perFrame + 2 * perFrame, 7 * perFrame);
            Assert.True(LongCatTestData.RelRms(noisyB, noisyA) > 1e-3, "the noisy frames should respond to their own latents.");
        }
    }

    [Fact]
    public void ConditioningFramesGetNoTextUpdate()
    {
        LongCatConditioning conditioning = new(NumCondLatents: 3);
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor textA = LongCatTestData.Encoder(_config, TextRows, 3);
        using Tensor textB = LongCatTestData.Encoder(_config, TextRows, 4);
        int perFrame = Height * Width;

        float[] a = Velocity(_weights, latent, textA, in conditioning);
        float[] b = Velocity(_weights, latent, textB, in conditioning);

        for (int c = 0; c < _config.OutChannels; c++)
            Assert.True(LongCatTestData.RelRms(b.AsSpan(c * Frames * perFrame, 3 * perFrame), a.AsSpan(c * Frames * perFrame, 3 * perFrame)) < 1e-6);
        Assert.True(LongCatTestData.RelRms(b, a) > 1e-3);
    }

    [Fact]
    public void ConditioningFramesTrackOnlyTheirOwnTimestep()
    {
        LongCatConditioning conditioning = new(NumCondLatents: 2);
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        int condLength = 2 * Height * Width;
        float[] atNoisy600 = Velocity(_weights, latent, text, in conditioning, timestep: 600f);
        float[] atNoisy300 = Velocity(_weights, latent, text, in conditioning, timestep: 300f);

        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        float[] steps = conditioning.FrameTimesteps(600f, Frames);
        steps[0] = 120f;
        using Tensor shifted = transformer.Forward(_backend, latent, text, steps, in conditioning);
        float[] shiftedValues = shifted.AsReadOnlySpan<float>().ToArray();

        for (int c = 0; c < _config.OutChannels; c++)
        {
            int start = c * Frames * Height * Width;
            Assert.True(LongCatTestData.RelRms(atNoisy300.AsSpan(start, condLength), atNoisy600.AsSpan(start, condLength)) < 1e-6,
                "a conditioning frame (timestep 0) moved when only the noisy timestep changed");
            Assert.True(LongCatTestData.RelRms(shiftedValues.AsSpan(start, condLength), atNoisy600.AsSpan(start, condLength)) > 1e-3,
                "a conditioning frame ignored its own timestep");
        }
        Assert.True(LongCatTestData.RelRms(atNoisy300, atNoisy600) > 1e-3);
    }

    [Fact]
    public void PerFrameTimestepsReachEachFramesOwnModulation()
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        float[] steps = new float[Frames];
        Array.Fill(steps, 500f);
        using Tensor uniform = transformer.Forward(_backend, latent, text, steps, LongCatConditioning.None);
        float[] uniformCopy = uniform.AsReadOnlySpan<float>().ToArray();
        using Tensor scalar = transformer.Forward(_backend, latent, text, [500f], LongCatConditioning.None);
        steps[4] = 40f;
        using Tensor varied = transformer.Forward(_backend, latent, text, steps, LongCatConditioning.None);

        Assert.True(LongCatTestData.RelRms(scalar.AsReadOnlySpan<float>(), uniformCopy) < 1e-7);
        Assert.True(LongCatTestData.RelRms(varied.AsReadOnlySpan<float>(), uniformCopy) > 1e-3);
    }

    [Fact]
    public void ReferenceFrameTakesItsOwnRopeIndexAndSkipWindow()
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        // 3 conditioning frames (reference + 2 context) of 9; noisy frames are 6.
        float[] plain = Velocity(_weights, latent, text, new LongCatConditioning(3, 1, 4, 0));
        float[] skipped = Velocity(_weights, latent, text, new LongCatConditioning(3, 1, 4, 1));
        float[] windowTooLate = Velocity(_weights, latent, text, new LongCatConditioning(3, 1, 10, 1));
        float[] windowTooLateNoRange = Velocity(_weights, latent, text, new LongCatConditioning(3, 1, 10, 0));
        float[] otherIndex = Velocity(_weights, latent, text, new LongCatConditioning(3, 1, 6, 0));

        Assert.True(LongCatTestData.RelRms(skipped, plain) > 1e-4, "hiding the reference from a noisy window changes those frames");
        Assert.Equal(windowTooLateNoRange, windowTooLate);
        Assert.True(LongCatTestData.RelRms(otherIndex, plain) > 1e-4, "the reference RoPE index reaches the attention");
        Assert.All(skipped, static v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void TextRowsPastTheValidLengthAreZeroedWhenPaddingIsKept()
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        float[] all = Velocity(_weights, latent, text, LongCatConditioning.None);
        float[] zeroPadded = Velocity(_weights, latent, text, LongCatConditioning.None, validText: 4);
        float[] fullyValid = Velocity(_weights, latent, text, LongCatConditioning.None, validText: TextRows);

        Assert.Equal(all, fullyValid);
        Assert.True(LongCatTestData.RelRms(zeroPadded, all) > 1e-4, "the padded rows are zeroed, so they stop carrying the caption");
        Assert.All(zeroPadded, static v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void DroppingPaddingEqualsAShorterCaption()
    {
        LongCatConfig dropping = _config with { TextTokensZeroPad = false };
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using Tensor shortText = text.SliceRows(0, 4);

        float[] dropped = Velocity(_weights, latent, text, LongCatConditioning.None, validText: 4, config: dropping);
        float[] shortened = Velocity(_weights, latent, shortText, LongCatConditioning.None, config: dropping);

        Assert.True(LongCatTestData.RelRms(dropped, shortened) < 1e-6);
    }

    [Fact]
    public void BlocksStreamAndEnumerateEveryWeightOnce()
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        IStreamableDenoiser streamable = transformer;

        Assert.Equal(_config.Depth, streamable.BlockCount);
        Assert.Same(streamable.GetBlock(0), streamable.GetBlock(0));
        Assert.All(Enumerable.Range(0, streamable.BlockCount), i => Assert.True(streamable.GetBlock(i).EstimatedWeightBytes > 0));
        List<Tensor> all = transformer.EnumerateWeights().ToList();
        Assert.Equal(all.Count, all.Distinct(ReferenceEqualityComparer.Instance).Count());
        int shared = streamable.EnumerateSharedWeights().Count();
        int blockWeights = Enumerable.Range(0, streamable.BlockCount).Sum(i => streamable.GetBlock(i).EnumerateWeights().Count());
        Assert.Equal(all.Count, shared + blockWeights);
        Assert.Equal(21, streamable.GetBlock(0).EnumerateWeights().Count());
    }

    private static void CaptureRope(Dictionary<string, float[]> stages, string name, Tensor table)
    {
        if (name.StartsWith("rope_", StringComparison.Ordinal))
            stages[name] = table.AsReadOnlySpan<float>().ToArray();
    }

    [Fact]
    public void RopeTablesFollowThePinnedFormula()
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        Dictionary<string, float[]> stages = [];
        using LongCatTransformer transformer = new(_config) { Probe = (name, t) => CaptureRope(stages, name, t) };
        transformer.LoadWeights(_weights);
        LongCatConditioning conditioning = new(2, 1, 5, 0);
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(600f, Frames), in conditioning);

        int gridH = Height / 2, gridW = Width / 2, headDim = _config.HeadDim;
        // head 32: t gets 32 - 4·(32/6) = 12 dims, h and w get 2·(32/6) = 10 each; frequencies 1/10000^(2j/dim), each repeated for its pair.
        int[] axisDims = [12, 10, 10];
        float[] cos = stages["rope_cos"], sin = stages["rope_sin"];
        Assert.Equal(Frames * gridH * gridW * headDim, cos.Length);
        int token = 0;
        for (int f = 0; f < Frames; f++)
            for (int h = 0; h < gridH; h++)
                for (int w = 0; w < gridW; w++, token++)
                {
                    float[] positions = [f == 0 ? 5f : f - 1, h, w];
                    int col = 0;
                    for (int axis = 0; axis < 3; axis++)
                        for (int j = 0; j < axisDims[axis] / 2; j++)
                        {
                            float frequency = (float)(1.0 / Math.Pow(10000.0, 2.0 * j / axisDims[axis]));
                            double angle = positions[axis] * frequency;
                            for (int pair = 0; pair < 2; pair++, col++)
                            {
                                Assert.True(Math.Abs(cos[token * headDim + col] - Math.Cos(angle)) < 2e-6, $"cos token {token} col {col}");
                                Assert.True(Math.Abs(sin[token * headDim + col] - Math.Sin(angle)) < 2e-6, $"sin token {token} col {col}");
                            }
                        }
                    Assert.Equal(headDim, col);
                }
    }
}
