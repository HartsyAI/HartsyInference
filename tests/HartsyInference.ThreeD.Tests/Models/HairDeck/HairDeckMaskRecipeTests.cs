using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.Vision.Segmentation;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

/// <summary>Instance selection of the HairDeck recipe on synthetic SAM 3 logits: 4 x 4 logit grids over an 8 x 8 working image,
/// each logit cell covering two working pixels. The synthetic matting model agrees with the person cells.</summary>
public sealed class HairDeckMaskRecipeTests
{
    private const int Height = 8;
    private const int Width = 8;
    private const int Side = 4;

    [Fact]
    public void Run_SelectsThePersonHairAndFaceOfOneCharacter()
    {
        TextGroundedLogitResult[] prompts =
        [
            Prompt("person", Instance(0.9f, Cells((r, c) => c >= 1))),
            Prompt("hair", Instance(0.9f, Cells((r, c) => r <= 1 && c is >= 1 and <= 2))),
            Prompt("face", Instance(0.9f, Cells((r, c) => r is >= 1 and <= 2 && c is >= 1 and <= 2))),
        ];

        HairDeckRecipeResult result = HairDeckMaskRecipe.Run(prompts, Height, Width, PersonMatte());

        Assert.Equal(HairDeckPrepStatus.Ok, result.Status);
        Assert.Equal([new HairDeckInstanceKey("hair", 0)], result.HairKeys);
        Assert.Equal([new HairDeckInstanceKey("face", 0)], result.FaceKeys);
        Assert.Equal(0, result.UnselectedPersonCount);
        Assert.False(result.BoxFallback);
        Assert.Equal(1.0, result.Agreement);
        Assert.True(result.Alpha[4 * Width + 4] > 0.5f);
        Assert.True(result.Alpha[4 * Width] < 0.5f);
    }

    [Fact]
    public void Run_WithoutAPersonAndWithoutTheMattingModelReportsNoPersonMask()
    {
        TextGroundedLogitResult[] prompts = [Prompt("hair", Instance(0.9f, Cells((r, c) => r <= 1)))];
        float[] matte = new float[Height * Width];
        Array.Fill(matte, 0.2f);

        HairDeckRecipeResult result = HairDeckMaskRecipe.Run(prompts, Height, Width, matte);

        Assert.Equal(HairDeckPrepStatus.NoPersonMask, result.Status);
        Assert.All(result.Alpha, value => Assert.Equal(1f, value));
        Assert.Single(result.HairKeys);
    }

    [Fact]
    public void Run_FallsBackToLowScoreFaceAndHairWhenNoneReachTheThreshold()
    {
        TextGroundedLogitResult[] prompts =
        [
            Prompt("person", Instance(0.9f, Cells((r, c) => c >= 1))),
            Prompt("hair", Instance(0.3f, Cells((r, c) => r <= 1))),
            Prompt("face", Instance(0.3f, Cells((r, c) => r is >= 1 and <= 2 && c is >= 1 and <= 2))),
        ];

        HairDeckRecipeResult result = HairDeckMaskRecipe.Run(prompts, Height, Width, PersonMatte());

        Assert.True(result.BoxFallback);
        Assert.Equal([new HairDeckInstanceKey("face", 0)], result.FaceKeys);
        Assert.Equal([new HairDeckInstanceKey("hair", 0)], result.HairKeys);
    }

    [Fact]
    public void Run_DropsHairInstancesMostlyOnFur()
    {
        TextGroundedLogitResult[] prompts =
        [
            Prompt("person", Instance(0.9f, Cells((r, c) => c >= 1))),
            Prompt("hair", Instance(0.9f, Cells((r, c) => r == 0 && c is >= 1 and <= 2)), Instance(0.8f, Cells((r, c) => r == 0 && c == 2))),
            Prompt("fur", Instance(0.9f, Cells((r, c) => r == 0 && c == 2))),
        ];

        HairDeckRecipeResult result = HairDeckMaskRecipe.Run(prompts, Height, Width, PersonMatte());

        Assert.Equal([new HairDeckInstanceKey("hair", 0)], result.HairKeys);
    }

    [Fact]
    public void Run_CountsPersonInstancesLeftOutOfTheCharacter()
    {
        TextGroundedLogitResult[] prompts =
        [
            Prompt("person", Instance(0.9f, Cells((r, c) => c >= 1)), Instance(0.8f, Cells((r, c) => r == 0 && c == 0))),
            Prompt("face", Instance(0.9f, Cells((r, c) => r is >= 1 and <= 2 && c is >= 1 and <= 2))),
        ];

        HairDeckRecipeResult result = HairDeckMaskRecipe.Run(prompts, Height, Width, PersonMatte());

        Assert.Equal(1, result.UnselectedPersonCount);
    }

    [Fact]
    public void Run_RejectsAMatteOfTheWrongSize()
    {
        Assert.Throws<ArgumentException>(() => HairDeckMaskRecipe.Run([], Height, Width, new float[10]));
    }

    /// <summary>Matte of the synthetic matting model: 0.9 on the person cells (columns 1 to 3), 0.1 elsewhere.</summary>
    private static float[] PersonMatte()
    {
        float[] matte = new float[Height * Width];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                matte[y * Width + x] = x / 2 >= 1 ? 0.9f : 0.1f;
            }
        }

        return matte;
    }

    /// <summary>A 4 x 4 logit grid: +10 where <paramref name="inside"/> holds, -10 elsewhere.</summary>
    private static float[] Cells(Func<int, int, bool> inside)
    {
        float[] logits = new float[Side * Side];
        for (int r = 0; r < Side; r++)
        {
            for (int c = 0; c < Side; c++)
            {
                logits[r * Side + c] = inside(r, c) ? 10f : -10f;
            }
        }

        return logits;
    }

    private static TextGroundedLogit Instance(float score, float[] logits) => new(score, logits);

    private static TextGroundedLogitResult Prompt(string prompt, params TextGroundedLogit[] instances)
        => new(prompt, Side, instances);
}
