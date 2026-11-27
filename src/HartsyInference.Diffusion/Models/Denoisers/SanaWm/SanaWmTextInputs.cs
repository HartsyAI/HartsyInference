using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM text conditioning. The CHI prefix is tokenized in front of the prompt, the padded sequence is reduced
/// to <see cref="MaxLength"/> positions (BOS plus the last 299), and the DiT consumes the selected hidden states and mask.
/// Tensors returned by the encode methods are owned by the caller.</summary>
public static class SanaWmTextInputs
{
    /// <summary>Text-encoder <c>model_max_length</c>: the number of conditioning positions the DiT consumes.</summary>
    public const int MaxLength = 300;

    /// <summary>The CHI prefix: the eight <c>text_encoder.chi_prompt</c> entries of the SANA-WM config joined by newlines.</summary>
    public const string ChiPrompt =
        "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions "
        + "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
        + "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and "
        + "spatial relationships to create vivid and concrete scenes.\n"
        + "- If the prompt is already detailed, refine and enhance the existing details slightly without "
        + "overcomplicating.\n"
        + "Here are examples of how to transform or refine prompts:\n"
        + "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round "
        + "shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red "
        + "flowers.\n"
        + "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
        + "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
        + "passing by towering glass skyscrapers.\n"
        + "Please generate only the enhanced description for the prompt below and avoid including any "
        + "additional commentary or evaluations:\n"
        + "User Prompt: ";

    private const int TailLength = MaxLength - 1;

    // Upstream sizes the padded CHI-prefixed sequence as the CHI token count plus this budget.
    private const int PromptTokenBudget = MaxLength - 2;

    /// <summary>Padded length of a CHI-prefixed prompt: the CHI token count, BOS included, plus <c>MaxLength - 2</c>.</summary>
    /// <param name="chiTokenCount">Token count of <see cref="ChiPrompt"/> alone, BOS included. Must be at least 2 so the
    /// padded length covers the selected positions.</param>
    public static int CondPaddedLength(int chiTokenCount)
    {
        if (chiTokenCount < 2)
            throw new ArgumentException("The CHI token count includes BOS and must be at least 2.", nameof(chiTokenCount));
        return chiTokenCount + PromptTokenBudget;
    }

    /// <summary>Source positions of the selected sequence: position 0, then the last 299 positions of the padded sequence in order.</summary>
    /// <param name="paddedLength">Length of the padded CHI-prefixed sequence; must be at least <see cref="MaxLength"/>.</param>
    public static int[] SelectIndices(int paddedLength)
    {
        if (paddedLength < MaxLength)
            throw new ArgumentException($"Padded length {paddedLength} is shorter than the {MaxLength} selected positions.", nameof(paddedLength));

        int[] indices = new int[MaxLength];
        int tailStart = paddedLength - TailLength;
        for (int i = 1; i < MaxLength; i++)
        {
            indices[i] = tailStart + i - 1;
        }
        return indices;
    }

    /// <summary>Token ids of <see cref="ChiPrompt"/> followed by a prompt, padded or truncated to the padded CHI length.</summary>
    /// <param name="promptIds">Tokenization of <c>ChiPrompt + prompt</c> as one string, BOS first.</param>
    /// <param name="chiTokenCount">Token count of <see cref="ChiPrompt"/> alone, BOS included.</param>
    public static int[] BuildCondIds(ReadOnlySpan<int> promptIds, int chiTokenCount)
    {
        ValidateBosFirst(promptIds, nameof(promptIds));
        return PadOrTruncate(promptIds, CondPaddedLength(chiTokenCount));
    }

    /// <summary>Token ids of a negative prompt with no CHI prefix, padded or truncated to <see cref="MaxLength"/>.</summary>
    /// <param name="negativeIds">Tokenization of the negative prompt alone, BOS first.</param>
    public static int[] BuildNegativeIds(ReadOnlySpan<int> negativeIds)
    {
        ValidateBosFirst(negativeIds, nameof(negativeIds));
        return PadOrTruncate(negativeIds, MaxLength);
    }

    /// <summary>Attention mask of the selected CHI-prefixed sequence: 1 where the selected position holds a real token.</summary>
    /// <param name="paddedLength">Length of the padded CHI-prefixed sequence.</param>
    /// <param name="validLength">Number of real tokens in the padded sequence, from 1 to <paramref name="paddedLength"/>.</param>
    /// <returns>An F32 tensor of shape <c>[1, 300]</c>. The caller owns it.</returns>
    public static Tensor SelectMask(int paddedLength, int validLength)
    {
        if (validLength < 1 || validLength > paddedLength)
            throw new ArgumentException($"Valid length {validLength} must be between 1 and the padded length {paddedLength}.", nameof(validLength));
        return CreateMask(SelectIndices(paddedLength), validLength);
    }

    /// <summary>Selects the conditioning positions of encoder hidden states <c>[1, L, C]</c>.</summary>
    /// <returns>An F32 tensor of shape <c>[1, 1, 300, C]</c>. The caller owns it.</returns>
    public static Tensor SelectHidden(IBackend backend, Tensor hidden)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hidden);
        ValidateHidden(hidden);
        return GatherHiddenRows(backend, hidden, SelectIndices(checked((int)hidden.Shape[1])));
    }

    /// <summary>Encodes a CHI-prefixed prompt into the DiT conditioning inputs.</summary>
    /// <param name="promptIds">Tokenization of <c>ChiPrompt + prompt</c> as one string, BOS first.</param>
    /// <param name="chiTokenCount">Token count of <see cref="ChiPrompt"/> alone, BOS included.</param>
    /// <returns>Cond <c>[1, 1, 300, C]</c> and mask <c>[1, 300]</c>. The caller disposes both tensors.</returns>
    public static (Tensor Cond, Tensor Mask) EncodeCondition(IBackend backend, LlamaStyleEncoder encoder,
        ReadOnlySpan<int> promptIds, int chiTokenCount)
    {
        ValidateEncoderArguments(backend, encoder);
        int[] ids = BuildCondIds(promptIds, chiTokenCount);
        int validLength = Math.Min(promptIds.Length, ids.Length);
        using Tensor hidden = EncodeRightPadded(backend, encoder, ids, validLength);
        Tensor cond = SelectHidden(backend, hidden);
        try
        {
            return (cond, SelectMask(ids.Length, validLength));
        }
        catch
        {
            cond.Dispose();
            throw;
        }
    }

    /// <summary>Encodes a prompt, tokenized with <paramref name="tokenizer"/> and prefixed with <see cref="ChiPrompt"/>.</summary>
    /// <returns>Cond <c>[1, 1, 300, C]</c> and mask <c>[1, 300]</c>. The caller disposes both tensors.</returns>
    public static (Tensor Cond, Tensor Mask) EncodeCondition(IBackend backend, LlamaStyleEncoder encoder,
        GemmaTokenizer tokenizer, string prompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(prompt);
        int[] promptIds = TokenizeWithBos(tokenizer, ChiPrompt + prompt);
        int chiTokenCount = TokenizeWithBos(tokenizer, ChiPrompt).Length;
        return EncodeCondition(backend, encoder, promptIds, chiTokenCount);
    }

    /// <summary>Encodes a negative prompt with no CHI prefix.</summary>
    /// <param name="negativeIds">Tokenization of the negative prompt alone, BOS first.</param>
    /// <returns>Cond <c>[1, 1, 300, C]</c> and mask <c>[1, 300]</c>. The caller disposes both tensors.</returns>
    public static (Tensor Cond, Tensor Mask) EncodeNegative(IBackend backend, LlamaStyleEncoder encoder, ReadOnlySpan<int> negativeIds)
    {
        ValidateEncoderArguments(backend, encoder);
        int[] ids = BuildNegativeIds(negativeIds);
        int validLength = Math.Min(negativeIds.Length, ids.Length);
        int[] positions = IdentityIndices(ids.Length);
        using Tensor hidden = EncodeRightPadded(backend, encoder, ids, validLength);
        Tensor cond = GatherHiddenRows(backend, hidden, positions);
        try
        {
            return (cond, CreateMask(positions, validLength));
        }
        catch
        {
            cond.Dispose();
            throw;
        }
    }

    /// <summary>Encodes a negative prompt, tokenized with <paramref name="tokenizer"/> and without the CHI prefix.</summary>
    /// <returns>Cond <c>[1, 1, 300, C]</c> and mask <c>[1, 300]</c>. The caller disposes both tensors.</returns>
    public static (Tensor Cond, Tensor Mask) EncodeNegative(IBackend backend, LlamaStyleEncoder encoder,
        GemmaTokenizer tokenizer, string negativePrompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(negativePrompt);
        return EncodeNegative(backend, encoder, TokenizeWithBos(tokenizer, negativePrompt));
    }

    private static int[] TokenizeWithBos(GemmaTokenizer tokenizer, string text)
    {
        IReadOnlyList<int> ids = tokenizer.EncodeRaw(text);
        // The SentencePiece tokenizer returns no ids for empty text, where the upstream tokenizer still emits BOS.
        if (ids.Count == 0)
            return [GemmaTokenizer.BosTokenId];
        return [.. ids];
    }

    private static Tensor EncodeRightPadded(IBackend backend, LlamaStyleEncoder encoder, int[] ids, int validLength)
    {
        int[][] batch = [ids];
        int[] layers = [encoder.NumLayers];
        int[] validLengths = [validLength];
        return encoder.EncodeMultiLayerRightPadded(backend, batch, layers, validLengths, normalizeFinalOutput: true);
    }

    private static Tensor GatherHiddenRows(IBackend backend, Tensor hidden, int[] positions)
    {
        long sequenceLength = hidden.Shape[1];
        long channels = hidden.Shape[2];
        Tensor output = new(new TensorShape(1, 1, positions.Length, channels), DType.F32);
        try
        {
            using Tensor rows = hidden.Reshape(new TensorShape(sequenceLength, channels));
            backend.GatherRows(output, rows, positions);
        }
        catch
        {
            output.Dispose();
            throw;
        }
        return output;
    }

    private static Tensor CreateMask(int[] positions, int validLength)
    {
        Tensor mask = new(new TensorShape(1, positions.Length), DType.F32);
        Span<float> values = mask.AsSpan<float>();
        for (int i = 0; i < positions.Length; i++)
        {
            values[i] = positions[i] < validLength ? 1f : 0f;
        }
        return mask;
    }

    private static int[] PadOrTruncate(ReadOnlySpan<int> ids, int length)
    {
        int[] result = new int[length];
        int count = Math.Min(ids.Length, length);
        ids[..count].CopyTo(result);
        result.AsSpan(count).Fill(GemmaTokenizer.PadTokenId);
        return result;
    }

    private static int[] IdentityIndices(int length)
    {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++)
        {
            indices[i] = i;
        }
        return indices;
    }

    private static void ValidateBosFirst(ReadOnlySpan<int> ids, string paramName)
    {
        if (ids.IsEmpty || ids[0] != GemmaTokenizer.BosTokenId)
            throw new ArgumentException($"Token ids must start with BOS ({GemmaTokenizer.BosTokenId}).", paramName);
    }

    private static void ValidateHidden(Tensor hidden)
    {
        if (hidden.Shape.Rank != 3 || hidden.Shape[0] != 1)
            throw new ArgumentException($"Encoder hidden states must be [1, L, C]; got {hidden.Shape}.", nameof(hidden));
        if (hidden.DType != DType.F32)
            throw new ArgumentException($"Encoder hidden states must be F32; got {hidden.DType.Name}.", nameof(hidden));
        if (hidden.Shape[1] < MaxLength)
            throw new ArgumentException($"Encoder hidden states hold {hidden.Shape[1]} positions; {MaxLength} are needed.", nameof(hidden));
    }

    private static void ValidateEncoderArguments(IBackend backend, LlamaStyleEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
    }
}
