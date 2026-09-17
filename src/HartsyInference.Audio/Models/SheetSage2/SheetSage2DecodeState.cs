using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>Everything one window's decode carries between steps: the cross-attention K/V, the self-attention
/// KV cache, and the working buffers the layers run in.
///
/// <para>Buffers are owned here rather than allocated per step. A transcription runs one step per token for up
/// to <see cref="SheetSage2Config.MaxTokens"/> tokens, so anything allocated inside the loop is allocated
/// thousands of times; <see cref="SheetSage2Scratch"/> is re-sized only when the chunk width changes, which
/// happens exactly once — from the prompt prefill to the single-token steps that follow.</para></summary>
public sealed class SheetSage2DecodeState : IDisposable
{
    private readonly SheetSage2Config _config;
    private int _disposed;

    internal SheetSage2DecodeState(SheetSage2Config config, int memoryTokens, int capacity, int tokenCount)
    {
        _config = config;
        MemoryTokens = memoryTokens;
        Capacity = capacity;
        CrossKey = new Tensor[config.Layers];
        CrossValue = new Tensor[config.Layers];
        SelfKey = new Tensor[config.Layers];
        SelfValue = new Tensor[config.Layers];
        TensorShape crossShape = new(1, config.NumHeads, memoryTokens, config.HeadDim);
        TensorShape selfShape = new(1, config.NumHeads, capacity, config.HeadDim);
        for (int i = 0; i < config.Layers; i++)
        {
            CrossKey[i] = new Tensor(crossShape, DType.F32);
            CrossValue[i] = new Tensor(crossShape, DType.F32);
            SelfKey[i] = new Tensor(selfShape, DType.F32);
            SelfValue[i] = new Tensor(selfShape, DType.F32);
        }
        Logits = new Tensor(new TensorShape(1, 1, tokenCount), DType.F32);
        Scratch = new SheetSage2Scratch(config);
    }

    /// <summary>Cross-attention keys per layer, head-major <c>[1, heads, memoryTokens, headDim]</c>, projected once.</summary>
    public Tensor[] CrossKey { get; }

    /// <summary>Cross-attention values per layer, head-major, projected once.</summary>
    public Tensor[] CrossValue { get; }

    /// <summary>Self-attention key cache per layer, <c>[1, heads, capacity, headDim]</c>; the first <see cref="Position"/> rows are valid.</summary>
    public Tensor[] SelfKey { get; }

    /// <summary>Self-attention value cache per layer.</summary>
    public Tensor[] SelfValue { get; }

    /// <summary>Logits for the most recent step, <c>[1, 1, tokenCount]</c>.</summary>
    public Tensor Logits { get; }

    /// <summary>Memory tokens the cross K/V hold.</summary>
    public int MemoryTokens { get; }

    /// <summary>Positions the self-attention cache can hold.</summary>
    public int Capacity { get; }

    /// <summary>Tokens written into the self-attention cache so far.</summary>
    public int Position { get; internal set; }

    internal SheetSage2Scratch Scratch { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _config.Layers; i++)
        {
            CrossKey[i].Dispose();
            CrossValue[i].Dispose();
            SelfKey[i].Dispose();
            SelfValue[i].Dispose();
        }
        Logits.Dispose();
        Scratch.Dispose();
    }
}

/// <summary>The working buffers one decoder layer runs in, sized to the current chunk width and reused by every
/// layer and every step at that width.</summary>
/// <remarks>Post-LN lets the buffers overlap tightly: the layer's input is dead the moment the first residual
/// sum is formed, so the final norm can write straight back over it and no ping-pong pair is needed.</remarks>
internal sealed class SheetSage2Scratch : IDisposable
{
    private readonly SheetSage2Config _config;
    private int _chunk;
    private int _disposed;

    public SheetSage2Scratch(SheetSage2Config config)
    {
        _config = config;
        LastHidden = new Tensor(new TensorShape(1, 1, config.Dim), DType.F32);
    }

    /// <summary>Token + position embeddings, before the embedding norm.</summary>
    public Tensor Embed { get; private set; } = null!;

    /// <summary>The block state flowing through the layers, <c>[1, chunk, dim]</c>.</summary>
    public Tensor Hidden { get; private set; } = null!;

    /// <summary>A residual sum awaiting its norm.</summary>
    public Tensor ResidualSum { get; private set; } = null!;

    /// <summary>The normalised block state between sub-blocks.</summary>
    public Tensor Normed { get; private set; } = null!;

    /// <summary>Self- or cross-attention queries, token-major.</summary>
    public Tensor Query { get; private set; } = null!;

    /// <summary>Self-attention keys for the new positions, token-major.</summary>
    public Tensor Key { get; private set; } = null!;

    /// <summary>Self-attention values for the new positions, token-major.</summary>
    public Tensor Value { get; private set; } = null!;

    /// <summary>Queries folded to <c>[1, heads, chunk, headDim]</c> for attention.</summary>
    public Tensor QueryHeads { get; private set; } = null!;

    /// <summary>New keys folded head-major, the shape the KV cache is appended in.</summary>
    public Tensor KeyHeads { get; private set; } = null!;

    /// <summary>New values folded head-major.</summary>
    public Tensor ValueHeads { get; private set; } = null!;

    /// <summary>Attention output, head-major.</summary>
    public Tensor Attention { get; private set; } = null!;

    /// <summary>Attention output folded back to <c>[1, chunk, dim]</c>.</summary>
    public Tensor AttentionFlat { get; private set; } = null!;

    /// <summary>Output of whichever projection closed the last sub-block.</summary>
    public Tensor Projected { get; private set; } = null!;

    /// <summary>Feed-forward inner activations before the GELU.</summary>
    public Tensor Feed { get; private set; } = null!;

    /// <summary>Feed-forward inner activations after the GELU.</summary>
    public Tensor FeedActivated { get; private set; } = null!;

    /// <summary>Causal mask for a multi-token prefill; null on the single-token path, which needs no mask.</summary>
    public Tensor? CausalMask { get; private set; }

    /// <summary>The last row of <see cref="Hidden"/> when a prefill produced several; the single-token path
    /// projects <see cref="Hidden"/> itself rather than copying it back from the device.</summary>
    public Tensor LastHidden { get; }

    /// <summary>Re-sizes the buffers when the chunk width changes; a no-op at the width already allocated.</summary>
    public void EnsureChunk(int chunk)
    {
        if (chunk == _chunk) return;
        Release();
        Resize(chunk);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Release();
        LastHidden.Dispose();
    }

    private void Resize(int chunk)
    {
        _chunk = chunk;
        TensorShape flat = new(1, chunk, _config.Dim);
        TensorShape heads = new(1, _config.NumHeads, chunk, _config.HeadDim);
        Embed = new Tensor(flat, DType.F32);
        Hidden = new Tensor(flat, DType.F32);
        ResidualSum = new Tensor(flat, DType.F32);
        Normed = new Tensor(flat, DType.F32);
        Query = new Tensor(flat, DType.F32);
        Key = new Tensor(flat, DType.F32);
        Value = new Tensor(flat, DType.F32);
        QueryHeads = new Tensor(heads, DType.F32);
        KeyHeads = new Tensor(heads, DType.F32);
        ValueHeads = new Tensor(heads, DType.F32);
        Attention = new Tensor(heads, DType.F32);
        AttentionFlat = new Tensor(flat, DType.F32);
        Projected = new Tensor(flat, DType.F32);
        Feed = new Tensor(new TensorShape(1, chunk, _config.IntermediateSize), DType.F32);
        FeedActivated = new Tensor(new TensorShape(1, chunk, _config.IntermediateSize), DType.F32);
        CausalMask = chunk > 1 ? new Tensor(new TensorShape(1, 1, chunk, chunk), DType.F32) : null;
    }

    private void Release()
    {
        if (_chunk == 0) return;
        Embed.Dispose();
        Hidden.Dispose();
        ResidualSum.Dispose();
        Normed.Dispose();
        Query.Dispose();
        Key.Dispose();
        Value.Dispose();
        QueryHeads.Dispose();
        KeyHeads.Dispose();
        ValueHeads.Dispose();
        Attention.Dispose();
        AttentionFlat.Dispose();
        Projected.Dispose();
        Feed.Dispose();
        FeedActivated.Dispose();
        CausalMask?.Dispose();
        CausalMask = null;
        _chunk = 0;
    }
}
