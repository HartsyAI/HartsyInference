using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM text inputs: the select rule, the padded layouts, and the hidden-state gather. The Integration test
/// compares the token ids with the upstream <c>text.safetensors</c> dump for the <c>A_demo_pose</c> run; it needs
/// <c>SANAWM_TEXT_REFERENCE</c> (that file) and <c>SANAWM_GEMMA2_TOKENIZER</c> (the Gemma-2 <c>tokenizer.model</c>).</summary>
public sealed class SanaWmTextInputsTests
{
    private const string ReferenceEnvironmentVariable = "SANAWM_TEXT_REFERENCE";
    private const string TokenizerEnvironmentVariable = "SANAWM_GEMMA2_TOKENIZER";
    private const int ReferenceChiTokenCount = 208;
    private const int ReferencePaddedLength = 506;
    private const int ReferenceValidLength = 379;
    private const int SyntheticChiBase = 1000;

    private const string ReferencePrompt =
        "A first-person view from a strictly stationary observation point across an immense dry lakebed "
        + "bordered by low mountain ranges. A black sports car occupies the central foreground on the "
        + "pale, compacted surface, aligned toward the open horizon beneath a vast blue sky. The "
        + "environment is broad and minimal, with flat beige desert crust, faint tire-worn texture, "
        + "distant rocky ridgelines, and a few thin clouds stretching across the upper sky. Bright midday "
        + "sunlight creates crisp shadows under the vehicle and a clean, high-visibility atmosphere of "
        + "speed, openness, and isolation. The observer's perspective remains fixed, with no dynamic "
        + "camera movement and no actions taken by the person recording. Autonomous motion belongs to the "
        + "world itself: dust trails sweep low across the ground, heat haze shimmers near the horizon, "
        + "clouds drift slowly, and the car's tires kick up fine desert grit.";

    [Theory]
    [InlineData(300)]
    [InlineData(506)]
    [InlineData(1024)]
    public void SelectIndices_KeepsPositionZeroAndTheLast299InOrder(int paddedLength)
    {
        int[] indices = SanaWmTextInputs.SelectIndices(paddedLength);

        Assert.Equal(SanaWmTextInputs.MaxLength, indices.Length);
        Assert.Equal(0, indices[0]);
        Assert.Equal(paddedLength - 299, indices[1]);
        Assert.Equal(paddedLength - 1, indices[^1]);
        for (int i = 2; i < indices.Length; i++)
        {
            Assert.Equal(indices[i - 1] + 1, indices[i]);
        }
    }

    [Fact]
    public void SelectIndices_ReferenceLength_StartsAtPosition207()
    {
        int[] indices = SanaWmTextInputs.SelectIndices(ReferencePaddedLength);

        Assert.Equal(207, indices[1]);
    }

    [Fact]
    public void SelectIndices_RejectsSequenceShorterThanMaxLength()
    {
        Assert.Throws<ArgumentException>(() => SanaWmTextInputs.SelectIndices(SanaWmTextInputs.MaxLength - 1));
    }

    [Fact]
    public void CondPaddedLength_IsChiTokenCountPlus298()
    {
        Assert.Equal(ReferencePaddedLength, SanaWmTextInputs.CondPaddedLength(ReferenceChiTokenCount));
    }

    [Fact]
    public void SelectMask_KeepsTheRealTokensOfTheSelectedPositions()
    {
        using Tensor mask = SanaWmTextInputs.SelectMask(ReferencePaddedLength, ReferenceValidLength);

        Assert.Equal(new TensorShape(1, SanaWmTextInputs.MaxLength), mask.Shape);
        ReadOnlySpan<float> values = mask.AsReadOnlySpan<float>();
        // Position 0 and source positions 207..378 are real: 173 ones in total, then padding.
        for (int i = 0; i < SanaWmTextInputs.MaxLength; i++)
        {
            Assert.Equal(i <= 172 ? 1f : 0f, values[i]);
        }
    }

    [Fact]
    public void SelectHidden_GathersTheSelectedRowsInOrder()
    {
        const int channels = 4;
        IBackend backend = new CpuBackend();
        using Tensor hidden = new(new TensorShape(1, ReferencePaddedLength, channels), DType.F32);
        Span<float> source = hidden.AsSpan<float>();
        for (int row = 0; row < ReferencePaddedLength; row++)
        {
            for (int col = 0; col < channels; col++)
            {
                source[row * channels + col] = row * 10f + col;
            }
        }

        using Tensor selected = SanaWmTextInputs.SelectHidden(backend, hidden);

        Assert.Equal(new TensorShape(1, 1, SanaWmTextInputs.MaxLength, channels), selected.Shape);
        int[] rows = SanaWmTextInputs.SelectIndices(ReferencePaddedLength);
        ReadOnlySpan<float> output = selected.AsReadOnlySpan<float>();
        for (int m = 0; m < SanaWmTextInputs.MaxLength; m++)
        {
            Assert.Equal(rows[m] * 10f, output[m * channels]);
            Assert.Equal(rows[m] * 10f + 3f, output[m * channels + 3]);
        }
    }

    [Fact]
    public void BuildCondIds_PadsTheJointIdsToTheChiLength()
    {
        int[] joint = SyntheticChiIds(ReferenceChiTokenCount, [5, 6]);

        int[] padded = SanaWmTextInputs.BuildCondIds(joint, ReferenceChiTokenCount);

        Assert.Equal(ReferencePaddedLength, padded.Length);
        Assert.Equal(joint, padded[..joint.Length]);
        Assert.All(padded[joint.Length..], id => Assert.Equal(GemmaTokenizer.PadTokenId, id));
    }

    [Fact]
    public void BuildNegativeIds_IsBosThenPromptThenPadding_WithoutChiPrefix()
    {
        int[] negative = [GemmaTokenizer.BosTokenId, 5, 6, 7];

        int[] padded = SanaWmTextInputs.BuildNegativeIds(negative);

        Assert.Equal(SanaWmTextInputs.MaxLength, padded.Length);
        Assert.Equal(negative, padded[..negative.Length]);
        Assert.All(padded[negative.Length..], id => Assert.Equal(GemmaTokenizer.PadTokenId, id));
        Assert.DoesNotContain(padded, id => id >= SyntheticChiBase);
    }

    [Fact]
    public void BuildNegativeIds_RejectsIdsWithoutBos()
    {
        int[] withoutBos = [5, 6];

        Assert.Throws<ArgumentException>(() => SanaWmTextInputs.BuildNegativeIds(withoutBos));
    }

    [Fact]
    public void ChiPrompt_IsTheEightConfigLinesJoinedByNewline()
    {
        // The checkpoint config carries eight chi_prompt entries; the reference length 506 = 208 + 298 confirms that count.
        string[] lines = SanaWmTextInputs.ChiPrompt.Split('\n');

        Assert.Equal(8, lines.Length);
        Assert.StartsWith("Given a user prompt, generate", lines[0]);
        Assert.Equal("User Prompt: ", lines[^1]);
        Assert.DoesNotContain(lines, static line => line.Length == 0);
    }

    [Trait("Category", "Integration")]
    [RealWeightsFact(ReferenceEnvironmentVariable, TokenizerEnvironmentVariable)]
    public void ReferenceIds_MatchTheUpstreamTokenizationAndSelection()
    {
        string referencePath = Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable)!;
        string tokenizerPath = Environment.GetEnvironmentVariable(TokenizerEnvironmentVariable)!;
        long[] condFull = ReadInt64Tensor(referencePath, "cond_ids_full");
        long[] condMaskFull = ReadInt64Tensor(referencePath, "cond_mask_full");
        long[] condSelected = ReadInt64Tensor(referencePath, "cond_ids_selected");
        long[] condMaskSelected = ReadInt64Tensor(referencePath, "cond_mask");
        long[] negIds = ReadInt64Tensor(referencePath, "neg_ids");

        using GemmaTokenizer tokenizer = new(tokenizerPath);
        int[] chiIds = [.. tokenizer.EncodeRaw(SanaWmTextInputs.ChiPrompt)];
        Assert.Equal(ReferenceChiTokenCount, chiIds.Length);
        Assert.Equal(ReferencePaddedLength, condFull.Length);

        int[] promptIds = [.. tokenizer.EncodeRaw(SanaWmTextInputs.ChiPrompt + ReferencePrompt)];
        Assert.Equal(ReferenceValidLength, promptIds.Length);
        int[] padded = SanaWmTextInputs.BuildCondIds(promptIds, chiIds.Length);
        AssertSameIds(condFull, padded, "cond_ids_full");

        int[] indices = SanaWmTextInputs.SelectIndices(padded.Length);
        int[] selected = new int[indices.Length];
        for (int m = 0; m < indices.Length; m++)
        {
            selected[m] = padded[indices[m]];
        }
        AssertSameIds(condSelected, selected, "cond_ids_selected");

        using Tensor mask = SanaWmTextInputs.SelectMask(padded.Length, promptIds.Length);
        ReadOnlySpan<float> maskValues = mask.AsReadOnlySpan<float>();
        for (int m = 0; m < maskValues.Length; m++)
        {
            Assert.Equal((float)condMaskSelected[m], maskValues[m]);
        }
        Assert.Equal(ReferenceValidLength, CountOnes(condMaskFull));

        // The reference negative prompt is empty: its ids are BOS alone (the BOS fallback for empty text in TokenizeWithBos).
        int[] negPromptIds = [GemmaTokenizer.BosTokenId];
        AssertSameIds(negIds, SanaWmTextInputs.BuildNegativeIds(negPromptIds), "neg_ids");
    }

    private static int[] SyntheticChiIds(int chiTokenCount, ReadOnlySpan<int> promptTail)
    {
        int[] ids = new int[chiTokenCount + promptTail.Length];
        ids[0] = GemmaTokenizer.BosTokenId;
        for (int i = 1; i < chiTokenCount; i++)
        {
            ids[i] = SyntheticChiBase + i;
        }
        promptTail.CopyTo(ids.AsSpan(chiTokenCount));
        return ids;
    }

    private static int CountOnes(long[] values)
    {
        int count = 0;
        foreach (long value in values)
        {
            if (value == 1)
            {
                count++;
            }
        }
        return count;
    }

    private static void AssertSameIds(long[] expected, int[] actual, string name)
    {
        Assert.True(expected.Length == actual.Length, $"{name}: expected {expected.Length} ids, got {actual.Length}.");
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i] == actual[i], $"{name}[{i}]: expected {expected[i]}, got {actual[i]}.");
        }
    }

    /// <summary>Reads one I64 tensor from a safetensors file. It parses the header itself, so the check does not depend on the loader under test.</summary>
    private static long[] ReadInt64Tensor(string path, string name)
    {
        byte[] file = File.ReadAllBytes(path);
        long headerLength = BitConverter.ToInt64(file, 0);
        using JsonDocument header = JsonDocument.Parse(file.AsMemory(8, checked((int)headerLength)));
        JsonElement entry = header.RootElement.GetProperty(name);
        Assert.Equal("I64", entry.GetProperty("dtype").GetString());
        long begin = entry.GetProperty("data_offsets")[0].GetInt64();
        long end = entry.GetProperty("data_offsets")[1].GetInt64();
        long[] values = new long[checked((int)((end - begin) / sizeof(long)))];
        Buffer.BlockCopy(file, checked((int)(8 + headerLength + begin)), values, 0, checked((int)(end - begin)));
        return values;
    }
}
