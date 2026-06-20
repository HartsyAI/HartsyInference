namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Configuration for the Moshi/Kyutai "depformer" — the depth transformer that autoregressively
/// predicts the <see cref="DepQ"/> Mimi codebooks for one temporal frame. It is RoPE-free (position is the
/// codebook index, captured by per-step weights) and uses <b>weights-per-step</b>: codebook step <c>k</c>
/// selects weight set <see cref="WeightSet"/>(k). Inputs are factorized via low-rank per-codebook
/// embeddings and a per-step ("multi-linear") input projection.</summary>
public sealed record MoshiDepthConfig
{
    public int Dim { get; init; } = 1_024;
    public int NumLayers { get; init; } = 4;
    public int NumHeads { get; init; } = 16;
    public int FfnDim { get; init; } = 3_072;
    public int DepQ { get; init; } = 32;
    public int CodebookVocab { get; init; } = 2_048;

    /// <summary>Temporal-backbone dim projected into the depformer as the step-0 context. 2048.</summary>
    public int MainDim { get; init; } = 2_048;

    /// <summary>Low-rank dimension of the per-codebook input embeddings.</summary>
    public int LowRankEmb { get; init; } = 128;

    public float RmsNormEps { get; init; } = 1e-8f;

    /// <summary>Per-codebook-step → weight-set index. Codebooks 0..7 each get a unique set (0..7);
    /// 8..15 → set 8; 16..23 → set 9; 24..31 → set 10. So <see cref="NumWeightSets"/> = 11.</summary>
    public IReadOnlyList<int> WeightSetSchedule { get; init; } = BuildSchedule(32);

    public int HeadDim => Dim / NumHeads;

    /// <summary>Number of distinct weight sets referenced by <see cref="WeightSetSchedule"/>.</summary>
    public int NumWeightSets
    {
        get
        {
            int max = 0;
            for (int i = 0; i < WeightSetSchedule.Count; i++) if (WeightSetSchedule[i] > max) max = WeightSetSchedule[i];
            return max + 1;
        }
    }

    /// <summary>Weight set used for codebook step <paramref name="k"/>.</summary>
    public int WeightSet(int k) => WeightSetSchedule[k];

    private static int[] BuildSchedule(int depQ)
    {
        int[] s = new int[depQ];
        for (int k = 0; k < depQ; k++)
            s[k] = k < 8 ? k : 8 + (k - 8) / 8;     // 0..7 unique, then blocks of 8 → 8,9,10
        return s;
    }

    public static MoshiDepthConfig Tts1_6B => new();
}
