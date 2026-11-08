using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Matting.BiRefNet;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam3;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Real-weight parity of the HairDeck preprocessing against the upstream hairdeck.prep / hairdeck.infer.image code,
/// run in float32 on the CPU. Needs <c>HAIRDECK_SAM3_PATH</c> (sam3.pt), <c>BIREFNET_MATTING_PATH</c> (model.safetensors),
/// <c>HAIRDECK_PREP_IMAGE</c> (the input PNG) and <c>HAIRDECK_PREP_REFERENCE</c> (the reference.safetensors written by
/// <c>tests/python-reference/dump_hairdeck_prep.py</c> for that image); reported as skipped when one is absent.</summary>
[Trait("Category", "Integration")]
public sealed class HairDeckPrepParityTests
{
    private const string Sam3Variable = "HAIRDECK_SAM3_PATH";
    private const string BiRefNetVariable = "BIREFNET_MATTING_PATH";
    private const string ImageVariable = "HAIRDECK_PREP_IMAGE";
    private const string ReferenceVariable = "HAIRDECK_PREP_REFERENCE";
    private static readonly string[] PromptOrder = ["person", "anime character", "hair", "face", "fur", "clothes", "hand"];

    [RealWeightsFact(Sam3Variable, BiRefNetVariable, ImageVariable, ReferenceVariable)]
    public void Preprocessing_MatchesTheUpstreamReference()
    {
        string sam3Path = Environment.GetEnvironmentVariable(Sam3Variable)!;
        string birefnetPath = Environment.GetEnvironmentVariable(BiRefNetVariable)!;
        string imagePath = Environment.GetEnvironmentVariable(ImageVariable)!;
        string referencePath = Environment.GetEnvironmentVariable(ReferenceVariable)!;

        using SafeTensorsLoader reference = new();
        reference.Load(referencePath);
        int height = (int)reference.GetTensor("person").Shape[0];
        int width = (int)reference.GetTensor("person").Shape[1];

        DecodedPng png = PngDecoder.DecodeWithAlpha(File.ReadAllBytes(imagePath));
        Assert.Equal(width, png.Width);
        Assert.Equal(height, png.Height);
        ReadOnlySpan<byte> referenceSource = Bytes(reference.GetTensor("rgba_src"));
        Assert.Equal(0, CountDifferent(referenceSource, Interleave(png.Rgb, png.Alpha!)));

        using IBackend backend = new CpuBackend();
        using Sam3Pipeline segmenter = Sam3Pipeline.Load(new Sam3Config(), sam3Path);
        // the converted weights reference the loader's memory, so the loader outlives the matter
        using SafeTensorsLoader weights = new();
        weights.Load(birefnetPath);
        using BiRefNetMatter matter = new();
        matter.LoadWeights(BiRefNetCheckpointConverter.Convert(weights.GetAllTensors()));

        HairDeckSourceImage source = HairDeckSourceImage.Create(png.Rgb, png.Alpha, png.Width, png.Height);
        HairDeckPreprocessor preprocessor = new(segmenter, matter, backend);
        HairDeckLoadedImage loaded = HairDeckImageRules.Load(source);
        IReadOnlyList<TextGroundedLogitResult> segmentation = preprocessor.Segment(loaded);
        float[] matte = preprocessor.MatteOf(loaded);
        HairDeckPreprocessResult ours = preprocessor.FromModels(loaded, segmentation, matte);

        CompareSegmentation(reference, segmentation);
        CompareMatte(reference, matte);
        ComparePreprocessing(reference, ours);
    }

    private void CompareSegmentation(SafeTensorsLoader reference, IReadOnlyList<TextGroundedLogitResult> ours)
    {
        for (int p = 0; p < PromptOrder.Length; p++)
        {
            ReadOnlySpan<float> refScores = Floats(reference.GetTensor($"sam_scores_{p}"));
            ReadOnlySpan<float> refLogits = Floats(reference.GetTensor($"sam_logits_{p}"));
            TextGroundedLogitResult theirs = ours[p];
            Assert.Equal(refScores.Length, theirs.Instances.Count);
            float scoreMax = 0f, logitMax = 0f;
            double logitSum = 0.0;
            for (int k = 0; k < refScores.Length; k++)
            {
                scoreMax = MathF.Max(scoreMax, MathF.Abs(refScores[k] - theirs.Instances[k].Score));
                for (int i = 0; i < theirs.LogitSide * theirs.LogitSide; i++)
                {
                    float diff = MathF.Abs(refLogits[k * theirs.LogitSide * theirs.LogitSide + i] - theirs.Instances[k].Logits[i]);
                    logitMax = MathF.Max(logitMax, diff);
                    logitSum += diff;
                }
            }

            Console.WriteLine($"sam3 {PromptOrder[p]}: {refScores.Length} instances, score maxAbs {scoreMax:G4}, "
                + $"logit maxAbs {logitMax:G4}, logit meanAbs {logitSum / Math.Max(1, refScores.Length * theirs.LogitSide * theirs.LogitSide):G4}");
            Assert.True(scoreMax < 1e-3f, $"{PromptOrder[p]}: score maxAbs {scoreMax}");
            Assert.True(logitMax < 1e-2f, $"{PromptOrder[p]}: logit maxAbs {logitMax}");
        }
    }

    private void CompareMatte(SafeTensorsLoader reference, float[] ours)
    {
        ReadOnlySpan<float> theirs = Floats(reference.GetTensor("alpha_full"));
        float max = 0f;
        double sum = 0.0;
        for (int i = 0; i < ours.Length; i++)
        {
            float diff = MathF.Abs(ours[i] - theirs[i]);
            max = MathF.Max(max, diff);
            sum += diff;
        }

        Console.WriteLine($"birefnet matte: maxAbs {max:G4}, meanAbs {sum / ours.Length:G4}");
        Assert.True(max < 0.05f, $"matte maxAbs {max}");
        Assert.True(sum / ours.Length < 0.002, $"matte meanAbs {sum / ours.Length}");
    }

    private void ComparePreprocessing(SafeTensorsLoader reference, HairDeckPreprocessResult ours)
    {
        ReadOnlySpan<byte> person = Bytes(reference.GetTensor("person"));
        ReadOnlySpan<byte> hair = Bytes(reference.GetTensor("hair"));
        ReadOnlySpan<byte> face = Bytes(reference.GetTensor("face"));
        int personMax = 0, personOver1 = 0;
        for (int i = 0; i < person.Length; i++)
        {
            int diff = Math.Abs(ours.Person[i] - person[i]);
            personMax = Math.Max(personMax, diff);
            personOver1 += diff > 1 ? 1 : 0;
        }

        int hairDiff = 0, faceDiff = 0;
        for (int i = 0; i < hair.Length; i++)
        {
            hairDiff += ours.Hair[i] == (hair[i] != 0) ? 0 : 1;
            faceDiff += ours.Face[i] == (face[i] != 0) ? 0 : 1;
        }

        ReadOnlySpan<float> box = Floats(reference.GetTensor("box"));
        double boxMax = Math.Max(
            Math.Max(Math.Abs(ours.Box.X0 - box[0]), Math.Abs(ours.Box.Y0 - box[1])),
            Math.Max(Math.Abs(ours.Box.X1 - box[2]), Math.Abs(ours.Box.Y1 - box[3])));

        ReadOnlySpan<byte> modelInput = Bytes(reference.GetTensor("model_input"));
        byte[] chw = ours.Input.Rgb448;
        int side = HairDeckInput.Resolution, plane = side * side;
        int inputMax = 0;
        double inputSum = 0.0;
        for (int i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                int diff = Math.Abs(chw[c * plane + i] - modelInput[i * 3 + c]);
                inputMax = Math.Max(inputMax, diff);
                inputSum += diff;
            }
        }

        ReadOnlySpan<byte> matteCrop = Bytes(reference.GetTensor("matte_crop"));
        int matteCropDiff = 0;
        for (int i = 0; i < plane; i++)
        {
            matteCropDiff += (ours.Input.Matte448[i] > 127) == (matteCrop[i] > 127) ? 0 : 1;
        }

        ReadOnlySpan<float> hairColor = Floats(reference.GetTensor("hair_color"));
        double colorMax = Math.Max(
            Math.Max(Math.Abs(ours.HairColor[0] - hairColor[0]), Math.Abs(ours.HairColor[1] - hairColor[1])),
            Math.Abs(ours.HairColor[2] - hairColor[2]));

        Console.WriteLine($"person maxAbs {personMax} LSB, {personOver1}/{person.Length} pixels off by more than 1 LSB");
        Console.WriteLine($"hair mismatches {hairDiff}, face mismatches {faceDiff} of {hair.Length} pixels");
        Console.WriteLine($"box {ours.Box} vs reference [{string.Join(", ", box.ToArray())}], maxAbs {boxMax:G4}");
        Console.WriteLine($"model input maxAbs {inputMax} LSB, meanAbs {inputSum / (3.0 * plane):G4}");
        Console.WriteLine($"matte crop mismatches {matteCropDiff} of {plane}; hair colour maxAbs {colorMax:G4}");
        Console.WriteLine($"status {ours.Status}, notes [{string.Join(", ", ours.Notes)}], agreement {ours.MatteAgreement}");

        Assert.Equal(HairDeckPrepStatus.Ok, ours.Status);
        Assert.True(boxMax <= 0.1 + 1e-4, $"head box differs by {boxMax}");
        Assert.True(personMax <= 2, $"person matte differs by {personMax} LSB");
        Assert.True(personOver1 <= person.Length / 1000, $"{personOver1} person pixels differ by more than 1 LSB");
        Assert.True(hairDiff <= person.Length / 500, $"{hairDiff} hair pixels differ");
        Assert.True(faceDiff <= person.Length / 500, $"{faceDiff} face pixels differ");
        Assert.True(inputMax <= 4, $"model input differs by {inputMax} LSB");
        Assert.True(matteCropDiff <= plane / 500, $"{matteCropDiff} matte crop pixels differ");
        Assert.True(colorMax < 0.01, $"hair colour differs by {colorMax}");
    }

    private static ReadOnlySpan<byte> Bytes(Tensor tensor) => tensor.AsReadOnlySpan<byte>();

    private static ReadOnlySpan<float> Floats(Tensor tensor) => tensor.AsReadOnlySpan<float>();

    private static byte[] Interleave(byte[] rgb, byte[] alpha)
    {
        byte[] rgba = new byte[alpha.Length * 4];
        for (int i = 0; i < alpha.Length; i++)
        {
            rgba[i * 4] = rgb[i * 3];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = alpha[i];
        }

        return rgba;
    }

    private static int CountDifferent(ReadOnlySpan<byte> expected, ReadOnlySpan<byte> actual)
    {
        int count = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            count += expected[i] == actual[i] ? 0 : 1;
        }

        return count;
    }
}
