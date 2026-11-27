using System.Text.Json;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>L0 parity: the packed layout and the causal mask against upstream inclusionAI/Duo at the commit recorded in <c>Fixtures/Duo/duo_l0_expected.json</c>,
/// produced by <c>tests/python-reference/duo_l0_oracle.py</c> from the cases in <c>duo_l0_cases.json</c>.</summary>
public sealed class DuoLayoutParityTests
{
    private const string CasesFile = "Fixtures/Duo/duo_l0_cases.json";
    private const string ExpectedFile = "Fixtures/Duo/duo_l0_expected.json";

    public static IEnumerable<object[]> Cases()
    {
        using JsonDocument expected = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        foreach (JsonElement c in expected.RootElement.GetProperty("cases").EnumerateArray())
        {
            yield return [c.GetProperty("name").GetString()!];
        }
    }

    [Fact]
    public void FixturesAreTheUpstreamRecord()
    {
        using JsonDocument expected = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        Assert.Equal("inclusionAI/Duo", expected.RootElement.GetProperty("source").GetString());
        Assert.Equal("068e357ede2bdea233b408645301e522f560da73", expected.RootElement.GetProperty("commit").GetString());
        Assert.Equal(5, expected.RootElement.GetProperty("cases").GetArrayLength());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void LayoutAndCausalMaskMatchUpstream(string name)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(CasesFile));
        JsonElement caseInput = document.RootElement.GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("name").GetString() == name);
        using JsonDocument expectedDocument = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        JsonElement expected = expectedDocument.RootElement.GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("name").GetString() == name);

        string rope = caseInput.GetProperty("rope").GetString()!;
        DuoLayout layout = DuoLayoutBuilder.Build(ReadSamples(caseInput), rope == "new" ? DuoRopePolicy.New : DuoRopePolicy.Old);

        Assert.Equal(expected.GetProperty("liveRows").GetInt32(), layout.LiveRows);
        Assert.Equal(expected.GetProperty("totalRows").GetInt32(), layout.Length);

        JsonElement groups = expected.GetProperty("groups");
        Assert.Equal(groups.GetArrayLength(), layout.Groups.Count);
        for (int i = 0; i < layout.Groups.Count; i++)
        {
            JsonElement g = groups[i];
            DuoGroup actual = layout.Groups[i];
            Assert.Equal(new DuoGroup(g[0].GetInt32(), g[1].GetInt32(), KindOf(g[2].GetString()!), ModalityOf(g[3]), g[4].GetInt32(), g[5].GetInt32()), actual);
        }

        double[] positions = Flatten(expected.GetProperty("positions"));
        Assert.Equal(positions.Length, layout.Positions.Length);
        for (int i = 0; i < positions.Length; i++)
        {
            Assert.InRange(layout.Positions[i], positions[i] - 1e-12, positions[i] + 1e-12);
        }
        Assert.Equal(expected.GetProperty("tags").EnumerateArray().Select(e => e.GetInt32()), layout.Tags);
        Assert.Equal(expected.GetProperty("timeIds").EnumerateArray().Select(e => e.GetInt32()), layout.TimeIds);

        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build(layout.Groups, layout.Groups, DuoAttentionMode.Causal);
        List<DuoMaskRect> expectedRects = expected.GetProperty("maskRects").EnumerateArray()
            .Select(r => new DuoMaskRect(r[0].GetInt32(), r[1].GetInt32(), r[2].GetInt32(), r[3].GetInt32())).ToList();
        Assert.Equal(expectedRects, rects);
    }

    [Fact]
    public void FlowGridMatchesUpstreamToOneF32Ulp()
    {
        using JsonDocument expectedDocument = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        foreach (JsonElement grid in expectedDocument.RootElement.GetProperty("grids").EnumerateArray())
        {
            int steps = grid.GetProperty("steps").GetInt32();
            double shift = grid.GetProperty("shift").GetDouble();
            float[] actual = DuoSchedule.Grid(steps, shift);
            float[] expected = grid.GetProperty("times").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.InRange(actual[i], expected[i] - 2e-7f, expected[i] + 2e-7f);
            }
        }
    }

    private static DuoGroupKind KindOf(string kind) => kind switch
    {
        "reference" => DuoGroupKind.Reference,
        "shared" => DuoGroupKind.Shared,
        "text" => DuoGroupKind.Text,
        "clean" => DuoGroupKind.Clean,
        "noisy" => DuoGroupKind.Noisy,
        "padding" => DuoGroupKind.Padding,
        _ => throw new InvalidDataException($"Unknown upstream group kind '{kind}'."),
    };

    private static DuoModality ModalityOf(JsonElement value) => value.ValueKind == JsonValueKind.Null ? DuoModality.None
        : value.GetString() switch
        {
            "video" => DuoModality.Video,
            "audio" => DuoModality.Audio,
            _ => throw new InvalidDataException("Unknown upstream modality."),
        };

    private static double[] Flatten(JsonElement rows)
    {
        List<double> values = [];
        foreach (JsonElement row in rows.EnumerateArray())
        {
            foreach (JsonElement v in row.EnumerateArray())
            {
                values.Add(v.GetDouble());
            }
        }
        return values.ToArray();
    }

    private static IReadOnlyList<DuoSampleInput> ReadSamples(JsonElement caseInput)
    {
        List<DuoSampleInput> samples = [];
        foreach (JsonElement s in caseInput.GetProperty("samples").EnumerateArray())
        {
            List<DuoBlockInput> blocks = [];
            foreach (JsonElement b in s.GetProperty("blocks").EnumerateArray())
            {
                JsonElement video = b.GetProperty("video");
                blocks.Add(new DuoBlockInput
                {
                    StartMs = b.GetProperty("startMs").GetDouble(),
                    BodyTextRows = b.GetProperty("bodyRows").GetInt32(),
                    Video = new DuoGrid(video[0].GetInt32(), video[1].GetInt32(), video[2].GetInt32()),
                    AudioSteps = b.GetProperty("audioSteps").GetInt32(),
                    FrameStartsMs = b.GetProperty("frameStartsMs").EnumerateArray().Select(v => v.GetDouble()).ToArray(),
                });
            }
            JsonElement reference = s.GetProperty("reference");
            samples.Add(new DuoSampleInput
            {
                SampleId = s.GetProperty("sampleId").GetInt32(),
                SharedDescriptionRows = s.GetProperty("sharedDescription").GetInt32(),
                SharedCueRows = s.GetProperty("sharedCue").GetInt32(),
                Reference = new DuoGrid(reference[0].GetInt32(), reference[1].GetInt32(), reference[2].GetInt32()),
                Blocks = blocks,
                CleanBlocks = s.GetProperty("clean").GetInt32(),
                NoisyBlocks = s.GetProperty("noisy").EnumerateArray().Select(v => v.GetInt32()).ToArray(),
            });
        }
        return samples;
    }
}
