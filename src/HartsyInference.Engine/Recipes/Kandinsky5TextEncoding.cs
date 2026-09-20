using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes;

/// <summary>Kandinsky 5's dual text-conditioning stack (Qwen2.5-VL-7B sequence embeddings + CLIP-L pooled), shared by
/// the T2I (<see cref="Image.Kandinsky5RecipePipeline"/>) and T2V (<see cref="Video.Kandinsky5VideoRecipePipeline"/>)
/// recipes — both take the same two embeddings from the reference <c>encode_prompt</c>, so the encode path is
/// identical regardless of which transformer/VAE consumes the output.</summary>
internal static unsafe class Kandinsky5TextEncoding
{
    /// <summary>Kandinsky's fixed conditioning system prompt — verbatim from diffusers <c>pipeline_kandinsky_t2i.py</c> (the "promt" typo is in the original).</summary>
    private const string SystemPrompt = "You are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

    /// <summary>Maximum prompt tokens after the template prefix (diffusers <c>max_sequence_length</c> default).</summary>
    private const int MaxPromptTokens = 512;

    /// <summary>Kandinsky 5's parity behaviour for <c>(text:N)</c>: take the grammar off the text and apply
    /// NOTHING. SwarmUI's probe puts this family on ComfyBlend because the CLIP-L arm keeps weights, but
    /// <c>Kandinsky5TEModel.encode_token_weights</c> (<c>kandinsky5.py:39-43</c>) returns the Qwen cond plus
    /// CLIP-L's POOLED vector and throws the blended hidden states away — so the weighting is a verified no-op
    /// end to end, and blending the Qwen arm to "fix" it would BREAK parity rather than achieve it.</summary>
    /// <remarks>Stripping still has to happen. Declaring the mode is what keeps <c>(text:N)</c> in the prompt
    /// through <c>ImagesService</c>'s flattening; without this the parens and digits would reach Qwen as prose,
    /// which is worse than the no-op it is supposed to reproduce.</remarks>
    internal static string StripEmphasis(string prompt) =>
        PromptWeighting.Join(PromptWeighting.Parse(prompt ?? ""));

    /// <summary>Runs the Qwen2.5-VL tower over the templated prompt and returns the last hidden state with the fixed template prefix sliced off — the reference takes <c>hidden_states[-1][:, prompt_template_encode_start_idx:]</c> (start idx 41 for the shipped template; computed here from the actual prefix ids so a tokenizer revision can't silently misalign it).</summary>
    internal static Tensor EncodeQwen(IBackend backend, LlamaStyleEncoder qwen, Qwen2Tokenizer tokenizer, string prompt)
    {
        prompt = StripEmphasis(prompt);
        int prefixLength = tokenizer.Encode($"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n", addSpecial: true).Length;
        int[] full = tokenizer.Encode($"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>", addSpecial: true);
        if (full.Length > prefixLength + MaxPromptTokens)
        {
            full = full[..(prefixLength + MaxPromptTokens)];
        }
        Tensor encoded = qwen.Encode(backend, new[] { full });
        try
        {
            return SliceSequenceFrom(encoded, prefixLength);
        }
        finally
        {
            encoded.Dispose();
        }
    }

    /// <summary>Runs CLIP-L over the 77-token prompt and returns the pooled EOS embedding <c>[1, 768]</c> (CLIPTextModel's <c>pooler_output</c> — CLIP-L has no text_projection, so the pooled value is the raw post-final-LN EOS hidden state).</summary>
    internal static Tensor EncodeClipPooled(IBackend backend, ClipTextEncoder clipL, ClipTokenizer tokenizer, string prompt)
    {
        int[] tokens = tokenizer.Encode(StripEmphasis(prompt));
        int eosPos = ClipTokenizer.FindEosPosition(tokens);
        (Tensor hidden, Tensor? pooled) = clipL.EncodePenultimate(backend, new[] { tokens }, new[] { eosPos }, layersFromEnd: 1);
        hidden.Dispose();
        if (pooled is null)
        {
            throw new InvalidOperationException("CLIP-L produced no pooled output; Kandinsky 5 requires the [1, 768] pooled embedding.");
        }
        return pooled;
    }

    /// <summary>Copies a <c>[batch, seq, hidden]</c> F32 tensor from <paramref name="start"/> to the end of the sequence axis, dropping the leading template tokens.</summary>
    private static Tensor SliceSequenceFrom(Tensor source, int start)
    {
        if (source.Shape.Rank != 3)
        {
            throw new ArgumentException($"Expected a rank-3 tensor, got rank {source.Shape.Rank}.", nameof(source));
        }
        if (source.DType != DType.F32)
        {
            throw new ArgumentException($"SliceSequenceFrom expects F32, got {source.DType}.", nameof(source));
        }
        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (start < 0 || start >= fullLen)
        {
            throw new ArgumentOutOfRangeException(nameof(start), $"start {start} out of range [0..{fullLen - 1}].");
        }
        long keep = fullLen - start;
        Tensor result = new Tensor(new TensorShape(batch, keep, hidden), source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long keepRowBytes = keep * hidden * elemSize;
        long offsetBytes = start * hidden * elemSize;
        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes + offsetBytes, dst + b * keepRowBytes, keepRowBytes, keepRowBytes);
        }
        return result;
    }
}
