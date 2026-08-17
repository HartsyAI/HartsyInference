using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Statics shared by the decoder-as-text-encoder implementations (<see cref="LlamaStyleEncoder"/>, <see cref="Gemma4TextEncoder"/>) so the two towers cannot drift apart on validation, RoPE table geometry, or tensor lifetime handling.</summary>
internal static unsafe class TextEncoderTensorHelpers
{
    /// <summary>Norm scales and embedding tables must be materialized as F32 to satisfy the backend gather/RMSNorm contracts; projection weights keep their checkpoint dtype.</summary>
    internal static Tensor CastToF32IfNeeded(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Cleanup must never replace a successful encoder result or mask the exception that caused an encode to unwind, so backend release failures are logged and swallowed.</summary>
    internal static void DisposeBestEffort(Tensor? tensor, string description)
    {
        if (tensor is null) return;
        try { tensor.Dispose(); }
        catch (Exception error) { Logs.Error($"Failed to release text encoder {description}.", error); }
    }

    /// <summary>Copies a small request constant through the backend once so every layer reuses its device-resident activation. A reshape/view is deliberately not used: views have a different tensor identity and therefore cannot find the producer's activation-cache entry.</summary>
    internal static Tensor MakeResidentCopy(IBackend backend, Tensor host)
    {
        Tensor resident = new(host.Shape, host.DType);
        try
        {
            backend.SliceRowsGeneric(resident, host, rowOffset: 0);
            return resident;
        }
        catch
        {
            DisposeBestEffort(resident, "resident request constant after copy failure");
            throw;
        }
        finally
        {
            DisposeBestEffort(host, "host request constant");
        }
    }

    /// <summary>Rejects ragged batches and out-of-vocabulary ids up front so bad prompts fail before any device allocation.</summary>
    internal static (int Batch, int Sequence) ValidateTokenBatch(int[][] tokenIds, int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Length == 0)
            throw new ArgumentException("Token batch must contain at least one row.", nameof(tokenIds));
        if (tokenIds[0] is null || tokenIds[0].Length == 0)
            throw new ArgumentException("Token rows must be non-null and non-empty.", nameof(tokenIds));

        int seqLen = tokenIds[0].Length;
        for (int b = 0; b < tokenIds.Length; b++)
        {
            int[]? row = tokenIds[b];
            if (row is null || row.Length != seqLen)
                throw new ArgumentException(
                    $"Token batch must be rectangular with sequence length {seqLen}; row {b} has length {row?.Length ?? 0}.",
                    nameof(tokenIds));
            for (int s = 0; s < row.Length; s++)
            {
                if ((uint)row[s] >= (uint)vocabSize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds), row[s], $"Token id at [{b},{s}] is outside vocabulary size {vocabSize}.");
            }
        }
        return (tokenIds.Length, seqLen);
    }

    /// <summary>Gathers token embeddings on the backend only when the vocabulary table is already device-resident: uploading a full table (262k rows for Gemma) solely to collect one prompt can cost multiple GB of VRAM, so otherwise this falls back to <see cref="EmbeddingLookupHost"/>.</summary>
    internal static Tensor EmbeddingLookup(IBackend backend, int[][] tokenIds, int batch, int seqLen,
        Tensor embedWeight, int hiddenSize, int vocabSize, float embeddingScale)
    {
        int rowCount = checked(batch * seqLen);
        int[] rows = new int[rowCount];
        int rowIndex = 0;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                rows[rowIndex++] = tokenIds[b][s];

        TensorShape shape = new(batch, seqLen, hiddenSize);
        Tensor gathered = new(shape, DType.F32);
        try
        {
            if (!backend.TryGatherRowsResident(gathered, embedWeight, rows))
            {
                DisposeBestEffort(gathered, "unused resident-gather output");
                return EmbeddingLookupHost(tokenIds, batch, seqLen, embedWeight, hiddenSize, vocabSize, embeddingScale);
            }
            if (embeddingScale == 1.0f)
                return gathered;

            Tensor scaled = new(shape, DType.F32);
            try
            {
                backend.Scale(scaled, gathered, embeddingScale);
            }
            catch
            {
                DisposeBestEffort(scaled, "scaled embedding output after failure");
                throw;
            }
            DisposeBestEffort(gathered, "unscaled gathered embedding");
            return scaled;
        }
        catch
        {
            DisposeBestEffort(gathered, "gathered embedding after failure");
            throw;
        }
    }

    /// <summary>Host lookup retained for multimodal/audio callers that immediately splice or slice the returned buffer on the CPU; uploading here only to read straight back would be strictly worse. The scale multiply is skipped when 1.0 (only Gemma-family encoders scale by sqrt(hidden)).</summary>
    internal static Tensor EmbeddingLookupHost(int[][] tokenIds, int batch, int seqLen,
        Tensor embedWeight, int hiddenSize, int vocabSize, float embeddingScale)
    {
        Tensor output = new(new TensorShape(batch, seqLen, hiddenSize), DType.F32);
        float* embedPtr = (float*)embedWeight.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = tokenIds[b][s];
                if ((uint)tokenId >= (uint)vocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds),
                        $"Token id {tokenId} at [{b},{s}] out of vocab (size {vocabSize}).");

                long src = (long)tokenId * hiddenSize;
                long dst = ((long)b * seqLen + s) * hiddenSize;
                Buffer.MemoryCopy(embedPtr + src, outPtr + dst, hiddenSize * sizeof(float), hiddenSize * sizeof(float));
            }
        }
        if (embeddingScale != 1.0f)
        {
            long total = (long)batch * seqLen * hiddenSize;
            for (long i = 0; i < total; i++) outPtr[i] *= embeddingScale;
        }
        return output;
    }

    /// <summary>Additive attention mask: query <c>i</c> may attend key <c>j</c> when <c>i - window &lt; j &lt;= i</c>; pass <see cref="int.MaxValue"/> for plain causal attention. Uses -1e30 rather than -inf to avoid NaN in mixed-precision softmax.</summary>
    internal static Tensor BuildCausalMask(int seqLen, int window)
    {
        const float negInf = -1e30f;
        Tensor mask = new(new TensorShape(seqLen, seqLen), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                bool blocked = j > i || (window != int.MaxValue && j <= i - window);
                p[(long)i * seqLen + j] = blocked ? negInf : 0f;
            }
        }
        return mask;
    }

    /// <summary>Expands a cached split-half table <c>[S, D/2]</c> to the backend RoPE contract <c>[B, S, D]</c>, duplicating each frequency into both vector halves and each batch row.</summary>
    internal static Tensor BuildFullWidthRopeTable(float[] halfTable, int batch, int seqLen, int headDim)
    {
        ArgumentNullException.ThrowIfNull(halfTable);
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be positive.");
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen), "Sequence length must be positive.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), "Split-half RoPE requires a positive even head dimension.");

        int half = headDim / 2;
        int required = checked(seqLen * half);
        if (halfTable.Length < required)
            throw new ArgumentException(
                $"RoPE table has {halfTable.Length} values, but sequence {seqLen} and head dimension {headDim} require at least {required}.",
                nameof(halfTable));

        Tensor full = new(new TensorShape(batch, seqLen, headDim), DType.F32);
        float* output = (float*)full.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int sourceBase = s * half;
                long destinationBase = ((long)b * seqLen + s) * headDim;
                for (int i = 0; i < half; i++)
                {
                    float value = halfTable[sourceBase + i];
                    output[destinationBase + i] = value;
                    output[destinationBase + half + i] = value;
                }
            }
        }
        return full;
    }

    /// <summary>Rounds the RoPE cache target up to a power of two so incrementally growing prompts rebuild the table O(log n) times, capped at the model's position limit.</summary>
    internal static int GrowRopeTableLength(int seqLen, int builtForLength, int maxPositionEmbeddings)
    {
        int target = Math.Max(seqLen, builtForLength);
        int rounded = 1;
        while (rounded < target) rounded <<= 1;
        return Math.Min(rounded, maxPositionEmbeddings);
    }

    /// <summary>Fills split-half cos/sin tables <c>[positions, half]</c>; angles stay in double so long-context positions keep phase precision, and zero frequencies yield identity rotation (partial rotary).</summary>
    internal static (float[] Cos, float[] Sin) BuildHalfRopeTable(double[] inverseFrequencies, int positions)
    {
        int half = inverseFrequencies.Length;
        float[] cos = new float[(long)positions * half];
        float[] sin = new float[(long)positions * half];
        for (int p = 0; p < positions; p++)
        {
            for (int k = 0; k < half; k++)
            {
                double angle = p * inverseFrequencies[k];
                cos[p * half + k] = (float)Math.Cos(angle);
                sin[p * half + k] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    /// <summary>All-ones F32 scale for weightless RMS norms — the backend RmsNorm contract always takes a weight tensor.</summary>
    internal static Tensor Ones(int length)
    {
        Tensor ones = new(new TensorShape(length), DType.F32);
        float* p = (float*)ones.DataPointer;
        for (int i = 0; i < length; i++) p[i] = 1f;
        return ones;
    }
}
