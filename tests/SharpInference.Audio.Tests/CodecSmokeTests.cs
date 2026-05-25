using SharpInference.Audio.Models.Codecs;
using SharpInference.Audio.Models.Codecs.BiCodec;
using SharpInference.Audio.Models.Codecs.Dac;
using SharpInference.Audio.Models.Codecs.EnCodec;
using SharpInference.Audio.Models.Codecs.Mimi;
using SharpInference.Audio.Models.Codecs.Snac;
using SharpInference.Audio.Models.Codecs.WavTokenizer;
using SharpInference.Audio.Models.Codecs.XCodec;
using SharpInference.Audio.Tests.Fixtures;
using SharpInference.Core.Tensors;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Codec smoke tests — generic and parameterized via
/// <see cref="CodecCatalog"/>. Adding a new codec is one new <see cref="CodecCatalog.Entry"/>;
/// these test methods auto-pick it up.
///
/// <para>Three universal checks per codec:</para>
/// <list type="number">
///   <item><b>Constructs cleanly</b> — instantiation without exceptions.</item>
///   <item><b>Sample rate matches the published checkpoint</b></item>
///   <item><b>Frame rate matches the published checkpoint</b></item>
///   <item><b>Max codebook count matches</b></item>
/// </list>
///
/// <para>Codec-specific config-shape tests (knobs that are unique per codec) stay as
/// targeted <c>[Fact]</c>s at the bottom of the file.</para></summary>
public sealed class CodecSmokeTests
{
    // ── Generic parameterized tests over the whole catalog ──────────────

    [Theory]
    [MemberData(nameof(CodecCatalog.Keys), MemberType = typeof(CodecCatalog))]
    public void Codec_ConstructsCleanly(string key)
    {
        CodecCatalog.Entry entry = CodecCatalog.Get(key);
        object codec = entry.Construct();
        Assert.NotNull(codec);
    }

    [Theory]
    [MemberData(nameof(CodecCatalog.Keys), MemberType = typeof(CodecCatalog))]
    public void Codec_SampleRateMatchesUpstream(string key)
    {
        CodecCatalog.Entry entry = CodecCatalog.Get(key);
        object codec = entry.Construct();
        Assert.Equal(entry.ExpectedSampleRate, entry.GetSampleRate(codec));
    }

    [Theory]
    [MemberData(nameof(CodecCatalog.Keys), MemberType = typeof(CodecCatalog))]
    public void Codec_FrameRateMatchesUpstream(string key)
    {
        CodecCatalog.Entry entry = CodecCatalog.Get(key);
        object codec = entry.Construct();
        Assert.Equal(entry.ExpectedFrameRate, entry.GetFrameRate(codec));
    }

    [Theory]
    [MemberData(nameof(CodecCatalog.Keys), MemberType = typeof(CodecCatalog))]
    public void Codec_CodebookCountMatchesUpstream(string key)
    {
        CodecCatalog.Entry entry = CodecCatalog.Get(key);
        object codec = entry.Construct();
        Assert.Equal(entry.ExpectedMaxCodebooks, entry.GetMaxCodebooks(codec));
    }

    // ── Codec-specific config knobs (not covered by the generic catalog) ─

    [Fact]
    public void EnCodec24kHz_ArchitecturalConstants()
    {
        EnCodecConfig c = EnCodecConfig.EnCodec24kHz;
        Assert.Equal(128, c.LatentDim);
        Assert.Equal(32, c.NFilters);
        Assert.Equal(new[] { 8, 5, 4, 2 }, c.Ratios.ToArray());
        Assert.Equal(2, c.LstmLayers);
        Assert.True(c.Causal);
        Assert.Equal(1_024, c.VqCodebookSize);
    }

    [Fact]
    public void Dac_AllVariantsShareArchitecture()
    {
        // All three sample-rate variants must agree on encoder dim, codebook geometry,
        // and the residual-unit dilation schedule.
        foreach (DacConfig c in new[] { DacConfig.Dac44kHz, DacConfig.Dac24kHz, DacConfig.Dac16kHz })
        {
            Assert.Equal(64, c.EncoderDim);
            Assert.Equal(1_024, c.CodebookSize);
            Assert.Equal(8, c.CodebookDim);
            Assert.Equal(1_024, c.LatentDim);
            Assert.Equal(new[] { 1, 3, 9 }, c.ResidualDilations.ToArray());
        }
    }

    [Fact]
    public void Snac_VqStridesAlwaysMatchCodebookCount()
    {
        foreach (SnacConfig c in new[] { SnacConfig.Snac24kHz, SnacConfig.Snac32kHz, SnacConfig.Snac44kHz })
            Assert.Equal(c.NCodebooks, c.VqStrides.Count);
    }

    [Fact]
    public void BiCodec_FsqVocabIsProductOfLevels()
    {
        BiCodecConfig c = BiCodecConfig.Default;
        int vocab = Fsq.VocabSize([.. c.GlobalFsqLevels]);
        Assert.Equal(8 * 8 * 8 * 5 * 5, vocab);
    }

    [Fact]
    public void Mimi_TransformerOfCodecsIsConfigured()
    {
        MimiConfig c = MimiConfig.Mimi24kHz;
        Assert.Equal(8, c.TransformerLayers);
        Assert.Equal(8, c.TransformerHeads);
        Assert.Equal(512, c.TransformerDim);
        Assert.Equal(7, c.AcousticCodebooks);
        Assert.Equal(8, c.TotalCodebooks);      // 1 semantic + 7 acoustic
    }

    // ── FSQ primitives ──────────────────────────────────────────────────

    [Fact]
    public void Fsq_VocabSize_IsProductOfLevels()
    {
        Assert.Equal(4_096, Fsq.VocabSize([8, 8, 8, 8]));
        Assert.Equal(12_800, Fsq.VocabSize([8, 8, 8, 5, 5]));
        Assert.Equal(1_024, Fsq.VocabSize([4, 4, 4, 4, 4]));
    }

    [Fact]
    public unsafe void Fsq_RoundTripIsStable()
    {
        int[] levels = [5, 5, 5, 5];
        int batch = 1, t = 4, d = 4;
        Tensor z = new(new TensorShape(batch, t, d), DType.F32);
        try
        {
            Random rng = new(42);
            float* zp = (float*)z.DataPointer;
            for (int i = 0; i < batch * t * d; i++) zp[i] = (float)(rng.NextDouble() * 2 - 1);

            using Tensor codes1 = new(new TensorShape(batch, t), DType.I32);
            using Tensor zHat = new(new TensorShape(batch, t, d), DType.F32);
            using Tensor codes2 = new(new TensorShape(batch, t), DType.I32);

            Fsq.Quantize(codes1, z, levels);
            Fsq.Dequantize(zHat, codes1, levels);
            Fsq.Quantize(codes2, zHat, levels);

            int* c1 = (int*)codes1.DataPointer;
            int* c2 = (int*)codes2.DataPointer;
            for (int i = 0; i < batch * t; i++) Assert.Equal(c1[i], c2[i]);
        }
        finally
        {
            z.Dispose();
        }
    }
}
