using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.BiCodec;
using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Codecs.Snac;
using HartsyInference.Audio.Models.Codecs.WavTokenizer;
using HartsyInference.Audio.Models.Codecs.XCodec;
using EnCodecModel = HartsyInference.Audio.Models.Codecs.EnCodec.EnCodec;
using DacModel = HartsyInference.Audio.Models.Codecs.Dac.Dac;
using SnacModel = HartsyInference.Audio.Models.Codecs.Snac.Snac;
using WavTokenizerModel = HartsyInference.Audio.Models.Codecs.WavTokenizer.WavTokenizer;
using BiCodecModel = HartsyInference.Audio.Models.Codecs.BiCodec.BiCodec;
using XCodecModel = HartsyInference.Audio.Models.Codecs.XCodec.XCodec;
using MimiModel = HartsyInference.Audio.Models.Codecs.Mimi.Mimi;

namespace HartsyInference.Audio.Tests.Fixtures;

/// <summary>Central registry of every neural audio codec we ship plus its expected
/// architectural constants. Tests iterate this catalog via <c>[Theory]</c> /
/// <c>[MemberData]</c> instead of one <c>[Fact]</c> per codec — adding a new codec
/// means one new entry here, not a new test class.
///
/// <para>Each <see cref="Entry"/> captures:</para>
/// <list type="bullet">
///   <item><b>Key</b> — stable string id used in xUnit test names</item>
///   <item><b>Construct</b> — factory that returns the codec instance</item>
///   <item><b>Probe</b> — lambdas pulling sample rate, frame rate, codebook count off
///         the instance (each codec has its own class, no shared interface)</item>
///   <item><b>Expected*</b> — known-good values matching the upstream Python checkpoint</item>
/// </list></summary>
public static class CodecCatalog
{
    public sealed record Entry(
        string Key,
        string Description,
        Func<object> Construct,
        Func<object, int> GetSampleRate,
        Func<object, int> GetFrameRate,
        Func<object, int> GetMaxCodebooks,
        int ExpectedSampleRate,
        int ExpectedFrameRate,
        int ExpectedMaxCodebooks);

    /// <summary>All codecs we test today. Adding a new codec = one new entry.</summary>
    public static readonly IReadOnlyList<Entry> All =
    [
        new(
            Key: "encodec_24khz",
            Description: "Meta EnCodec 24 kHz mono (Bark / MusicGen / AudioGen)",
            Construct: () => new EnCodecModel(EnCodecConfig.EnCodec24kHz),
            GetSampleRate: o => ((EnCodecModel)o).SampleRate,
            GetFrameRate: o => ((EnCodecModel)o).FrameRate,
            GetMaxCodebooks: o => ((EnCodecModel)o).MaxCodebooks,
            ExpectedSampleRate: 24_000,
            ExpectedFrameRate: 75,
            ExpectedMaxCodebooks: 32),
        new(
            Key: "dac_44khz",
            Description: "Descript Audio Codec 44.1 kHz (IndexTTS / Higgs Audio acoustic)",
            Construct: () => new DacModel(DacConfig.Dac44kHz),
            GetSampleRate: o => ((DacModel)o).SampleRate,
            GetFrameRate: o => ((DacModel)o).FrameRate,
            GetMaxCodebooks: o => ((DacModel)o).MaxCodebooks,
            ExpectedSampleRate: 44_100,
            ExpectedFrameRate: 86,
            ExpectedMaxCodebooks: 9),
        new(
            Key: "dac_24khz",
            Description: "Descript Audio Codec 24 kHz",
            Construct: () => new DacModel(DacConfig.Dac24kHz),
            GetSampleRate: o => ((DacModel)o).SampleRate,
            GetFrameRate: o => ((DacModel)o).FrameRate,
            GetMaxCodebooks: o => ((DacModel)o).MaxCodebooks,
            ExpectedSampleRate: 24_000,
            ExpectedFrameRate: 75,
            ExpectedMaxCodebooks: 32),
        new(
            Key: "dac_16khz",
            Description: "Descript Audio Codec 16 kHz (Spark-TTS HiFi-GAN wavegen)",
            Construct: () => new DacModel(DacConfig.Dac16kHz),
            GetSampleRate: o => ((DacModel)o).SampleRate,
            GetFrameRate: o => ((DacModel)o).FrameRate,
            GetMaxCodebooks: o => ((DacModel)o).MaxCodebooks,
            ExpectedSampleRate: 16_000,
            ExpectedFrameRate: 50,
            ExpectedMaxCodebooks: 12),
        new(
            Key: "snac_24khz",
            Description: "SNAC 24 kHz hierarchical RVQ (Orpheus TTS)",
            Construct: () => new SnacModel(SnacConfig.Snac24kHz),
            GetSampleRate: o => ((SnacModel)o).SampleRate,
            GetFrameRate: o => ((SnacModel)o).FrameRate,
            GetMaxCodebooks: o => ((SnacModel)o).NCodebooks,
            ExpectedSampleRate: 24_000,
            ExpectedFrameRate: 46,           // 24000 / (2*4*8*8) = 46.875 ≈ 46 with int division
            ExpectedMaxCodebooks: 4),
        new(
            Key: "snac_32khz",
            Description: "SNAC 32 kHz",
            Construct: () => new SnacModel(SnacConfig.Snac32kHz),
            GetSampleRate: o => ((SnacModel)o).SampleRate,
            GetFrameRate: o => ((SnacModel)o).FrameRate,
            GetMaxCodebooks: o => ((SnacModel)o).NCodebooks,
            ExpectedSampleRate: 32_000,
            ExpectedFrameRate: 62,           // 32000 / 512 = 62.5 → 62
            ExpectedMaxCodebooks: 4),
        new(
            Key: "snac_44khz",
            Description: "SNAC 44.1 kHz (music-focused)",
            Construct: () => new SnacModel(SnacConfig.Snac44kHz),
            GetSampleRate: o => ((SnacModel)o).SampleRate,
            GetFrameRate: o => ((SnacModel)o).FrameRate,
            GetMaxCodebooks: o => ((SnacModel)o).NCodebooks,
            ExpectedSampleRate: 44_100,
            ExpectedFrameRate: 100,          // 44100 / (3*3*7*7) = 100
            ExpectedMaxCodebooks: 3),
        new(
            Key: "wavtokenizer_24khz",
            Description: "WavTokenizer 24 kHz single-codebook",
            Construct: () => new WavTokenizerModel(WavTokenizerConfig.WavTokenizer24kHz),
            GetSampleRate: o => ((WavTokenizerModel)o).SampleRate,
            GetFrameRate: o => ((WavTokenizerModel)o).FrameRate,
            GetMaxCodebooks: _ => 1,
            ExpectedSampleRate: 24_000,
            ExpectedFrameRate: 75,
            ExpectedMaxCodebooks: 1),
        new(
            Key: "bicodec_default",
            Description: "Spark-TTS BiCodec (semantic + global FSQ tokenizer)",
            Construct: () => new BiCodecModel(BiCodecConfig.Default),
            GetSampleRate: _ => 0,           // BiCodec consumes w2v-BERT features, not raw audio
            GetFrameRate: _ => 50,           // semantic stream is 50 Hz
            GetMaxCodebooks: _ => 2,         // 1 semantic VQ + 1 global FSQ stream
            ExpectedSampleRate: 0,
            ExpectedFrameRate: 50,
            ExpectedMaxCodebooks: 2),
        new(
            Key: "xcodec_16khz",
            Description: "XCodec 16 kHz (YuE music)",
            Construct: () => new XCodecModel(XCodecConfig.XCodec16kHz),
            GetSampleRate: o => ((XCodecModel)o).SampleRate,
            GetFrameRate: o => ((XCodecModel)o).FrameRate,
            GetMaxCodebooks: o => ((XCodecModel)o).NCodebooks,
            ExpectedSampleRate: 16_000,
            ExpectedFrameRate: 50,
            ExpectedMaxCodebooks: 8),
        new(
            Key: "mimi_24khz",
            Description: "Kyutai Mimi (Sesame CSM / Moshi)",
            Construct: () => new MimiModel(MimiConfig.Mimi24kHz),
            GetSampleRate: o => ((MimiModel)o).SampleRate,
            GetFrameRate: o => ((MimiModel)o).FrameRate,
            GetMaxCodebooks: o => ((MimiModel)o).NCodebooks,
            ExpectedSampleRate: 24_000,
            ExpectedFrameRate: 25,           // 24000 / (8*6*5*4) = 25
            ExpectedMaxCodebooks: 8),        // 1 semantic + 7 acoustic
    ];

    /// <summary>xUnit MemberData feed — yields one row per catalog entry suitable for
    /// <c>[Theory]</c>. Use as <c>[MemberData(nameof(CodecCatalog.Keys), MemberType = typeof(CodecCatalog))]</c>.</summary>
    public static IEnumerable<object[]> Keys => All.Select(e => new object[] { e.Key });

    public static Entry Get(string key) => All.First(e => e.Key == key);
}
