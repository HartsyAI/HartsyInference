using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Reading upstream's int8 tables out of <c>rnnoise_data.c</c>.
///
/// <para>The Integration case is the proof that the file holds what upstream compiles. Its digests are SHA-256s of
/// the arrays as gcc lays them out in memory, taken from the pinned tarball's <c>rnnoise_data.c</c> by a program that
/// includes the file and writes each array's bytes:</para>
/// <code>
///   #include "rnnoise_data.c"
///   #define DUMP(a) dump(#a, a, sizeof(a))   /* fwrite each array to &lt;name&gt;.bin */
///   gcc -O0 -DDISABLE_DEBUG_FLOAT -Isrc -Iinclude dump.c src/parse_lpcnet_weights.c
/// </code>
/// <para>run in xiph/rnnoise at 70f1d25 with the tarball's <c>src/</c> files in place.</para></summary>
public sealed class RnnoiseInt8TablesTests(ITestOutputHelper log)
{
    /// <summary>SHA-256 of each array's bytes as gcc compiles <c>rnnoise_data.c</c> from the pinned tarball.</summary>
    private static readonly Dictionary<string, string> GccDigests = new(StringComparer.Ordinal)
    {
        ["conv2_weights_int8"] = "3cf607de3667c8996571cc1b163051f1ccd9a95064ea4b649adcf49f08e58b6e",
        ["conv2_scale"] = "995098c4e278980c2476d1652e48998904035c056f9581714b2d3a085c3dcd41",
        ["conv2_subias"] = "d82a6f6d1dfdbf56f2dd9dea5cb1afb35a03f1e0c6b6f433fb96efe2652b10c8",
        ["gru1_input_weights_int8"] = "6ae439c99193663f31829700dc58664ac0706b6fa74c01468f3b06daa37f8d9f",
        ["gru1_input_scale"] = "82594d08eecbb78195741f401e1ab0a8b1f5c8a2267a147fd80677678a42747c",
        ["gru1_input_subias"] = "d4eabc3fa8286124ed2a3557639550f68f567beffbcd05b29694f10fc3c42257",
        ["gru1_recurrent_weights_int8"] = "88af1dcbffe1c974aa2b3cffb0a9621ea7a3a9373adb79ea5a0bcc7212fbf41d",
        ["gru1_recurrent_scale"] = "9443f316cd34ab223a880e266e4ffa9dc668bde784fa95fa53f6a248813a9795",
        ["gru1_recurrent_subias"] = "4eca9189090243b627fe91e51009448a31969e5a7afa04c9a1096b48ab0355da",
        ["gru1_recurrent_weights_diag"] = "095958c5297baf76e1fa8665e5d36521ee344f624d43a6dbe716e8a90c254c13",
        ["gru2_input_weights_int8"] = "49e7be13d98c3b62bb716af494ce71d2c55459d8ab5f94e99175847e23b55408",
        ["gru2_input_scale"] = "b950737d2927998e63b497dffe50a27c8e51dbe5b651fb414239062787d20772",
        ["gru2_input_subias"] = "5c5197bd4bd920205140e3dd28deaa6b5f57c0e7e0d203257328e67f43030834",
        ["gru2_recurrent_weights_int8"] = "8c9824a11529300c0d8ebed1b5e1df14ae18f5e4bf4b0c395074ebd09da97ae1",
        ["gru2_recurrent_scale"] = "1d601b15df216b0b25fb49f50658a8f784cc289882bc33043d6a3b5311d8f3f8",
        ["gru2_recurrent_subias"] = "b6abbb1a19e0d10a8f0c3d4e638a5958cc2620c0516d77d4e8ea355966dc9fd7",
        ["gru2_recurrent_weights_diag"] = "deb29f07a4a3c78a7d78fa0f3d3e0ed12a573d06c2f35e86bc60390e7b800104",
        ["gru3_input_weights_int8"] = "a6cdadb2d5e5a152ff41b35c79a0606e7171abaa3aa8f495ad05255a03c58713",
        ["gru3_input_scale"] = "e8439dff51116a8d4da21594cddaa9e47ec4303a54bed16ccfb19564225ea261",
        ["gru3_input_subias"] = "47739c0f2e1305f8d1aedd0671b93d11d24a44870cf2722a4c9cbe629bcbe755",
        ["gru3_recurrent_weights_int8"] = "72f83c2805f2ae51e0ee3691962971845a0a94a6ada67febfe28d3fe2fa3adab",
        ["gru3_recurrent_scale"] = "0c7383a3fdeebc653993c1e287f3fb7fa9889fe62dd6204652f893c8f53bf61a",
        ["gru3_recurrent_subias"] = "2902ba26ebc11f2d1693d75115497642387b43bd7a8e8d10f11602808262d2bd",
        ["gru3_recurrent_weights_diag"] = "c8d915b7db1a9fb213b5b4e4818ba6ba9f46efdcb84786405a08f7055ca900e1",
    };

    [Fact]
    [Trait("Category", "Integration")]
    public void ConvertTarball_RealXiphTarball_IsByteEqualToWhatGccCompiles()
    {
        string? tarball = Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_TARBALL");
        if (tarball is null)
        {
            log.WriteLine("SKIPPED: set HARTSYINFERENCE_RNNOISE_TARBALL to xiph's rnnoise_data-*.tar.gz");
            return;
        }
        if (!RealWeightGate.Require(log.WriteLine, tarball)) return;

        string directory = Directory.CreateTempSubdirectory("rnnoise-int8-").FullName;
        try
        {
            string output = Path.Combine(directory, RnnoiseInt8Tables.FileName);
            RnnoiseInt8Tables.ConvertTarball(tarball, output);
            using SafeTensorsLoader loader = new();
            loader.Load(output);
            Assert.Equal(GccDigests.Keys.Order(), loader.Descriptors.Keys.Order());
            foreach ((string name, string digest) in GccDigests)
            {
                using Tensor tensor = loader.GetTensor(name);
                Assert.Equal(digest, Convert.ToHexString(SHA256.HashData(tensor.AsSpan<byte>())).ToLowerInvariant());
            }
            log.WriteLine($"{GccDigests.Count} arrays byte-equal to gcc's; {new FileInfo(output).Length} bytes");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Parse_ReadsTheInt8Arrays_AsACompilerWould_AndSkipsEveryOtherArray()
    {
        Source source = Source.Build(seed: 1);
        Dictionary<string, Tensor> tensors = RnnoiseInt8Tables.Parse(new StringReader(source.Text));
        try
        {
            Assert.Equal(RnnoiseInt8Tables.Arrays.Keys.Order(), tensors.Keys.Order());
            foreach ((string name, Array values) in source.Values)
            {
                Tensor tensor = tensors[name];
                if (values is sbyte[] int8) Assert.True(tensor.AsSpan<sbyte>().SequenceEqual(int8), $"'{name}' differs");
                else Assert.True(tensor.AsSpan<float>().SequenceEqual((float[])values), $"'{name}' differs");
            }
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Parse_RefusesASparseIndexList()
    {
        // Drop block 0's last column group: the count says 95 and the list stops one short.
        string text = Source.Build(seed: 2).Text.Replace(
            "static const int gru2_recurrent_weights_idx[13968] = {\n    96, 0,",
            "static const int gru2_recurrent_weights_idx[13968] = {\n    95, 0,", StringComparison.Ordinal);
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => RnnoiseInt8Tables.Parse(new StringReader(text)));
        Assert.Contains("gru2_recurrent_weights_idx", error.Message);
    }

    [Fact]
    public void Parse_RefusesAnArrayOfAnotherLength()
    {
        string text = Source.Build(seed: 3).Text.Replace("conv2_scale[384]", "conv2_scale[385]", StringComparison.Ordinal);
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => RnnoiseInt8Tables.Parse(new StringReader(text)));
        Assert.Contains("conv2_scale", error.Message);
    }

    /// <summary>An <c>rnnoise_data.c</c> in upstream's format: every array the tables need at its real length,
    /// dense index lists, and float debug copies and an unrelated array the parser must skip.</summary>
    private sealed class Source
    {
        public required string Text { get; init; }
        public required Dictionary<string, Array> Values { get; init; }

        public static Source Build(int seed)
        {
            Random rng = new(seed);
            StringBuilder text = new("/* generated */\n#include \"rnnoise_data.h\"\n\n");
            Dictionary<string, Array> values = new(StringComparer.Ordinal);
            AppendFloats(text, "conv1_weights_float", Floats(rng, 24));
            foreach ((string name, int count) in RnnoiseInt8Tables.Arrays)
            {
                if (name.EndsWith("_weights_int8", StringComparison.Ordinal))
                {
                    sbyte[] weights = new sbyte[count];
                    for (int i = 0; i < count; i++) weights[i] = (sbyte)rng.Next(-128, 128);
                    values[name] = weights;
                    text.Append("#ifndef USE_WEIGHTS_FILE\n");
                    Append(text, "opus_int8", name, weights.Select(v => v.ToString(CultureInfo.InvariantCulture)));
                    text.Append("#endif /* USE_WEIGHTS_FILE */\n\n#ifndef DISABLE_DEBUG_FLOAT\n");
                    AppendFloats(text, name.Replace("_int8", "_float", StringComparison.Ordinal), Floats(rng, 16));
                    text.Append("#endif /*DISABLE_DEBUG_FLOAT*/\n\n");
                    if (name.StartsWith("gru", StringComparison.Ordinal))
                        Append(text, "int", name.Replace("_int8", "_idx", StringComparison.Ordinal), DenseIndex(1152, 384));
                }
                else
                {
                    float[] floats = Floats(rng, count);
                    values[name] = floats;
                    AppendFloats(text, name, floats);
                }
            }
            return new Source { Text = text.ToString(), Values = values };
        }

        private static IEnumerable<string> DenseIndex(int outputs, int inputs)
        {
            for (int block = 0; block < outputs / 8; block++)
            {
                yield return (inputs / 4).ToString(CultureInfo.InvariantCulture);
                for (int column = 0; column < inputs; column += 4) yield return column.ToString(CultureInfo.InvariantCulture);
            }
        }

        private static float[] Floats(Random rng, int count)
        {
            float[] floats = new float[count];
            for (int i = 0; i < count; i++) floats[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, -rng.Next(0, 6)));
            return floats;
        }

        // Python's repr of the float widened to double, as upstream's exporter prints: parsing that as a double and
        // rounding to float gives the float back.
        private static void AppendFloats(StringBuilder text, string name, float[] floats) =>
            Append(text, "float", name, floats.Select(v => ((double)v).ToString("R", CultureInfo.InvariantCulture)));

        private static void Append(StringBuilder text, string type, string name, IEnumerable<string> items)
        {
            string[] all = [.. items];
            text.Append($"static const {type} {name}[{all.Length}] = {{\n");
            for (int i = 0; i < all.Length; i += 8)
                text.Append("    ").Append(string.Join(", ", all.Skip(i).Take(8))).Append(",\n");
            text.Append("};\n\n");
        }
    }
}
