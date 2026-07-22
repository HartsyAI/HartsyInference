using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Microsoft Lens's text-encoder front-end — a thin wrapper around <see cref="GptOssEncoder"/>
/// that hardcodes the canonical 24-layer GPT-OSS preset, the four selected hidden-state layers
/// (<c>[5, 11, 17, 23]</c>), and the chat-template <c>txt_offset</c> stripping. Use this rather than
/// instantiating <see cref="GptOssEncoder"/> directly when the consumer is the Lens diffusion DiT —
/// it guarantees the indices, ordering, and offset semantics match upstream
/// <c>lens/pipeline.py:LensPipeline._get_text_embeddings</c> exactly.</summary>
public sealed class LensGptOssEncoder : IDisposable
{
    /// <summary>Decoder-layer indices captured by the Lens pipeline. Hardcoded — these are baked into
    /// how the DiT was trained, do not expose as a runtime knob.</summary>
    public static readonly int[] SelectedLayers = [5, 11, 17, 23];

    /// <summary>Tokens stripped from the front of every encoder output. Matches
    /// <c>lens/pipeline.py:DEFAULT_TXT_OFFSET</c>. The chat template Lens prepends (system prompt +
    /// assistant-thinking wrapper) always tokenizes to exactly 97 tokens.</summary>
    public const int DefaultTextOffset = 97;

    private readonly GptOssEncoder _encoder;

    /// <summary>Constructs the encoder with the canonical Lens preset
    /// (<see cref="LlamaStyleEncoderConfig.GptOss"/>).</summary>
    public LensGptOssEncoder()
        : this(LlamaStyleEncoderConfig.GptOss)
    {
    }

    /// <summary>Constructs the encoder from a custom config (e.g. a future Lens variant). The config
    /// must still satisfy GPT-OSS structural requirements (MoE on, layer-types of length NumLayers,
    /// sinks on) — see <see cref="GptOssEncoder"/>.</summary>
    public LensGptOssEncoder(LlamaStyleEncoderConfig config)
    {
        _encoder = new GptOssEncoder(config);
    }

    /// <summary>Number of transformer layers (24 for the canonical preset).</summary>
    public int NumLayers => _encoder.NumLayers;

    /// <summary>Loads encoder weights from a HuggingFace-format state dict (keys
    /// <c>model.embed_tokens.weight</c>, <c>model.layers.{i}.*</c>, <c>model.norm.weight</c>). MoE
    /// experts arrive either dense F32 (the diffusers MXFP4 path dequantizes at load via
    /// <see cref="HartsyInference.ModelAssets.Mxfp4.Mxfp4Codec.DequantGptOssExpertsInPlace"/>) or
    /// NVFP4-packed (the ComfyUI path — kept quantized and mmap-backed; <see cref="GptOssMoeFfn"/>
    /// streams one expert at a time at forward time so the 20B bank never materializes at F32). Both
    /// <c>LensCheckpointConverter</c> outputs are ready to pass here directly.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights) => _encoder.LoadWeights(weights);

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights();

    /// <summary>Encodes a single chat-templated prompt (already token-ID encoded) and returns the four
    /// multi-layer hidden-state captures expected by <see cref="HartsyInference.Diffusion.Pipelines.LensPipeline"/>.
    /// Strips the first <see cref="DefaultTextOffset"/> tokens from each capture to drop the system prompt
    /// wrapper, matching upstream's <c>_get_text_embeddings</c>.
    /// <para>Output: <c>List&lt;Tensor&gt;</c> of length 4. Each tensor has shape <c>[1, txtSeqLen, 2880]</c>
    /// where <c>txtSeqLen = tokenIds.Length - DefaultTextOffset</c> (the user-prompt-and-prefix portion
    /// of the hidden state). Returns four empty <c>[1, 0, 2880]</c> tensors when the input is shorter
    /// than the offset.</para></summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="tokenIds">Chat-templated token ids (rendered by <see cref="HartsyInference.ModelAssets.Tokenizers.GptOssTokenizer.BuildChatInputs"/>).</param>
    public List<Tensor> EncodeForLens(IBackend backend, int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Length <= DefaultTextOffset)
            throw new ArgumentException(
                $"Prompt tokenization yielded only {tokenIds.Length} tokens; the chat-template wrapper alone " +
                $"requires {DefaultTextOffset + 1}+ tokens before user content begins. Make sure to render " +
                $"through GptOssTokenizer.BuildChatInputs, which prepends the system prompt + assistant " +
                $"thinking block.",
                nameof(tokenIds));

        int[][] batched = [tokenIds];
        List<Tensor> raw = _encoder.EncodeAtLayers(backend, batched, SelectedLayers);
        try
        {
            List<Tensor> stripped = new(raw.Count);
            for (int i = 0; i < raw.Count; i++)
                stripped.Add(StripLeading(raw[i], DefaultTextOffset));
            return stripped;
        }
        finally
        {
            for (int i = 0; i < raw.Count; i++) raw[i].Dispose();
        }
    }

    /// <summary>Slices a <c>[B, S, H]</c> tensor to drop the first <paramref name="offset"/> sequence
    /// positions. The caller is responsible for ensuring <paramref name="offset"/> &lt; <c>S</c> — this
    /// helper is invoked only after the public entry point has validated the input.</summary>
    private static unsafe Tensor StripLeading(Tensor t, int offset)
    {
        int batch = (int)t.Shape[0];
        int seqLen = (int)t.Shape[1];
        int hidden = (int)t.Shape[2];
        int newLen = seqLen - offset;
        Tensor sliced = new Tensor(new TensorShape(batch, newLen, hidden), DType.F32);
        float* srcPtr = (float*)t.DataPointer;
        float* dstPtr = (float*)sliced.DataPointer;
        long bytesPerToken = (long)hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            float* srcRow = srcPtr + ((long)b * seqLen + offset) * hidden;
            float* dstRow = dstPtr + (long)b * newLen * hidden;
            Buffer.MemoryCopy(srcRow, dstRow, (long)newLen * bytesPerToken, (long)newLen * bytesPerToken);
        }
        return sliced;
    }

    public void Dispose() => _encoder.Dispose();
}
