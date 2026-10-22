namespace HartsyInference.LLM.Tests.Attention;

/// <summary>The sequences <see cref="AttentionWaveE4TraceTests"/> pins, recorded from the code before the sites moved onto the shared layer.</summary>
internal static class AttentionWaveE4TracePins
{
    public static readonly Dictionary<string, string[]> All = new()
    {
        ["t5model_encode"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 8,2,8",
            "Permute0213 8,2,8",
            "Permute0213 8,2,8",
            "Attn Dense scale=1 f16=False out=[1, 2, 8, 8]:F32 q=[1, 2, 8, 8]:F32 k=[1, 2, 8, 8]:F32 v=[1, 2, 8, 8]:F32 mask=[1, 2, 8, 8]:F32 live=6",
            "Permute0213 2,8,8",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "Gelu",
            "Linear w24",
            "Mul",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "attention #0 after run: out=freed q=freed k=freed v=freed mask=freed",
        ],
        ["t5model_decode"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 3,2,8",
            "Permute0213 3,2,8",
            "Permute0213 3,2,8",
            "Attn Dense scale=1 f16=False out=[1, 2, 3, 8]:F32 q=[1, 2, 3, 8]:F32 k=[1, 2, 3, 8]:F32 v=[1, 2, 3, 8]:F32 mask=[1, 2, 3, 3]:F32 live=6",
            "Permute0213 2,3,8",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 3,2,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Attn Dense scale=1 f16=False out=[1, 2, 3, 8]:F32 q=[1, 2, 3, 8]:F32 k=[1, 2, 6, 8]:F32 v=[1, 2, 6, 8]:F32 mask=[1, 2, 3, 6]:F32 live=7",
            "Permute0213 2,3,8",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "Gelu",
            "Linear w24",
            "Mul",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w20",
            "attention #0 after run: out=freed q=freed k=freed v=freed mask=freed",
            "attention #1 after run: out=freed q=freed k=freed v=freed mask=freed",
        ],
    };
}
