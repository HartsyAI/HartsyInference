using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Produces the Kandinsky 5.0 dual text-encoder conditioning in-engine, replacing the
/// pre-computed-embeddings "diffusers escape hatch" the <see cref="Denoisers.Kandinsky5Transformer"/>
/// pipelines consumed. Wraps the reusable Qwen2.5-VL text tower (<see cref="LlamaStyleEncoder"/> with
/// <see cref="LlamaStyleEncoderConfig.Qwen2_5_VL_7B"/>) and CLIP-L (<see cref="ClipTextEncoder"/> with
/// <see cref="ClipTextEncoderConfig.SdxlClipL"/>), reproducing diffusers'
/// <c>Kandinsky5Pipeline._encode_prompt_qwen</c> / <c>_encode_prompt_clip</c> exactly:
/// <list type="bullet">
///   <item>Qwen: wrap the prompt in <see cref="QwenTemplate"/>, take <c>hidden_states[-1]</c> (the final
///   <c>model.norm</c> output), and drop the leading <see cref="QwenStartIdx"/> system-preamble rows →
///   <c>[1, seq, 3584]</c>.</item>
///   <item>CLIP: tokenize to <see cref="ClipMaxLength"/> (pad with EOT), take the final-LayerNorm'd hidden
///   state at the EOS position (un-projected — matches <c>CLIPTextModel.pooler_output</c>) → <c>[1, 768]</c>.</item>
/// </list>
/// Both outputs are returned as host F32 tensors, identical in layout to the <c>.bin</c> embeddings the
/// end-to-end tests load, so <c>Kandinsky5*Pipeline.GenerateFromEmbeddings</c> is unchanged.</summary>
public sealed unsafe class Kandinsky5PromptEncoder : IDisposable
{
    /// <summary>Qwen2.5-VL prompt wrapper, verbatim from diffusers <c>pipeline_kandinsky_t2i.py</c>
    /// (<c>prompt_template</c>). The "promt" spelling is a typo in the reference and must be reproduced
    /// exactly — it changes the tokenization and therefore the <see cref="QwenStartIdx"/> drop count.
    /// <c>{0}</c> is the user prompt.</summary>
    public const string QwenTemplate =
        "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, "
        + "size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        + "<|im_start|>user\n{0}<|im_end|>";

    /// <summary>Number of leading (system-preamble) hidden-state rows dropped from the Qwen output —
    /// <c>prompt_template_encode_start_idx</c> in the reference. Equals the token count of the templated
    /// prefix up to and including <c>&lt;|im_start|&gt;user\n</c>.</summary>
    public const int QwenStartIdx = 41;

    /// <summary>Max real (post-drop) Qwen sequence length; the templated input is truncated to
    /// <see cref="QwenStartIdx"/> + this.</summary>
    public const int QwenMaxSeq = 512;

    /// <summary>CLIP context length (<c>max_length=77</c>, <c>padding="max_length"</c>).</summary>
    public const int ClipMaxLength = 77;

    private readonly LlamaStyleEncoder _qwen;
    private readonly ClipTextEncoder _clip;
    private readonly Qwen2Tokenizer _qwenTok;
    private readonly ClipTokenizer _clipTok;
    private readonly bool _ownsEncoders;
    private bool _disposed;

    /// <param name="ownsEncoders">When true, <see cref="Dispose"/> also disposes the Qwen encoder and
    /// tokenizers (the CLIP encoder holds no unmanaged handles). Set false when the caller (e.g. a shared
    /// loader cache) owns the encoders' lifetime.</param>
    public Kandinsky5PromptEncoder(LlamaStyleEncoder qwen, ClipTextEncoder clip,
        Qwen2Tokenizer qwenTok, ClipTokenizer clipTok, bool ownsEncoders = false)
    {
        _qwen = qwen ?? throw new ArgumentNullException(nameof(qwen));
        _clip = clip ?? throw new ArgumentNullException(nameof(clip));
        _qwenTok = qwenTok ?? throw new ArgumentNullException(nameof(qwenTok));
        _clipTok = clipTok ?? throw new ArgumentNullException(nameof(clipTok));
        _ownsEncoders = ownsEncoders;
    }

    /// <summary>Encodes one prompt into <c>(qwenEmbeds [1, seq, 3584], clipPooled [1, 768])</c> host F32
    /// tensors ready for <c>GenerateFromEmbeddings</c>. Caller disposes both. An empty/whitespace prompt is
    /// valid (the reference default negative prompt is <c>""</c>).</summary>
    public (Tensor qwenEmbeds, Tensor clipPooled) Encode(IBackend backend, string prompt)
    {
        ThrowIfDisposed();
        prompt ??= "";

        Tensor qwenEmbeds = EncodeQwen(backend, prompt);
        Tensor clipPooled = EncodeClip(backend, prompt);
        return (qwenEmbeds, clipPooled);
    }

    private Tensor EncodeQwen(IBackend backend, string prompt)
    {
        string full = string.Format(QwenTemplate, prompt);
        int[] ids = _qwenTok.Encode(full, addSpecial: true);
        int maxTotal = QwenStartIdx + QwenMaxSeq;
        if (ids.Length > maxTotal)
            ids = ids[..maxTotal];
        if (ids.Length <= QwenStartIdx)
            throw new InvalidOperationException(
                $"Kandinsky5 Qwen tokenization produced {ids.Length} tokens, at or below the {QwenStartIdx}-row " +
                "system-preamble drop — the prompt template or tokenizer is misconfigured.");

        Tensor hidden = _qwen.Encode(backend, [ids]);           // [1, S, 3584] (post model.norm)
        try { return DropLeadingRows(hidden, QwenStartIdx); }   // [1, S - 41, 3584]
        finally { hidden.Dispose(); }
    }

    private Tensor EncodeClip(IBackend backend, string prompt)
    {
        int[] ids = _clipTok.Encode(prompt);                    // [77], EOT-padded
        int eosPos = ClipTokenizer.FindEosPosition(ids);
        if (eosPos < 0)
            throw new InvalidOperationException("Kandinsky5 CLIP tokenization produced no EOT token.");

        Tensor hidden = _clip.Encode(backend, new[] { ids }, layersFromEnd: 1); // [1, 77, 768], final-LN'd
        try { return ExtractRow(hidden, eosPos); }                              // [1, 768]
        finally { hidden.Dispose(); }
    }

    /// <summary>Copies rows <c>[start..]</c> of a <c>[1, S, H]</c> tensor into a fresh host F32
    /// <c>[1, S-start, H]</c>. Runs once per prompt (not a hot path); the <see cref="Tensor.DataPointer"/>
    /// read syncs the source to host, matching the layout the pre-computed <c>.bin</c> loaders produce.</summary>
    private static Tensor DropLeadingRows(Tensor hidden, int start)
    {
        int s = (int)hidden.Shape[1];
        int h = (int)hidden.Shape[2];
        int keep = s - start;
        Tensor result = new Tensor(new TensorShape(1, keep, h), DType.F32);
        float* src = (float*)hidden.DataPointer;
        float* dst = (float*)result.DataPointer;
        long bytes = (long)keep * h * sizeof(float);
        Buffer.MemoryCopy(src + (long)start * h, dst, bytes, bytes);
        return result;
    }

    /// <summary>Copies row <paramref name="row"/> of a <c>[1, S, H]</c> tensor into a fresh host F32
    /// <c>[1, H]</c>.</summary>
    private static Tensor ExtractRow(Tensor hidden, int row)
    {
        int h = (int)hidden.Shape[2];
        Tensor result = new Tensor(new TensorShape(1, h), DType.F32);
        float* src = (float*)hidden.DataPointer;
        float* dst = (float*)result.DataPointer;
        long bytes = (long)h * sizeof(float);
        Buffer.MemoryCopy(src + (long)row * h, dst, bytes, bytes);
        return result;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(Kandinsky5PromptEncoder));
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_ownsEncoders)
        {
            _qwen.Dispose();
            _qwenTok.Dispose();
            _clipTok.Dispose();
        }
    }
}
