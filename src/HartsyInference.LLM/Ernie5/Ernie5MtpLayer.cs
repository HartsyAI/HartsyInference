using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>One ERNIE-5.5 multi-token-prediction layer (vLLM-ERNIE <c>Ernie5_5MTPLayer</c>): it mixes the target's pre-head residual stream
/// with the next token's embedding, <c>h_proj(hnorm(h)) + e_proj(enorm(e))</c> on each of the <c>hc</c> streams, and runs a full decoder block
/// (<see cref="Ernie5Layer"/>) on the result. Its read-out (its own mHC head and <c>shared_head.norm</c>, then the target's LM head) is small
/// and stays with <see cref="Ernie5Model"/>, so a non-resident model need not rebuild this layer to read a draft.</summary>
internal sealed class Ernie5MtpLayer : IDisposable
{
    private readonly IBackend _backend;
    private readonly Ernie5Config _cfg;
    private readonly Tensor _embedNorm;
    private readonly Tensor _hiddenNorm;
    private readonly Tensor _embedProj;
    private readonly Tensor _hiddenProj;
    private readonly Ernie5Layer _block;
    private int _disposed;

    /// <summary>Reads MTP layer <paramref name="index"/>'s weights from <paramref name="source"/> and builds its block; a failure disposes what it
    /// read.</summary>
    public Ernie5MtpLayer(IBackend backend, Ernie5Config cfg, int index, IErnie5WeightSource source, IRopeRowSource hcaRope, IRopeRowSource dsaRope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(source);
        if (!cfg.IsMtpLayer(index)) throw new ArgumentOutOfRangeException(nameof(index), index, "Not a multi-token-prediction layer.");
        _backend = backend;
        _cfg = cfg;
        Index = index;
        int d = cfg.HiddenSize;
        try
        {
            _embedNorm = Ernie5Layer.ReadF32(source, Ernie5WeightNames.Mtp(index, Ernie5WeightNames.MtpEmbedNorm));
            _hiddenNorm = Ernie5Layer.ReadF32(source, Ernie5WeightNames.Mtp(index, Ernie5WeightNames.MtpHiddenNorm));
            _embedProj = ReadMatrix(source, Ernie5WeightNames.Mtp(index, Ernie5WeightNames.MtpEmbedProj), d, d);
            _hiddenProj = ReadMatrix(source, Ernie5WeightNames.Mtp(index, Ernie5WeightNames.MtpHiddenProj), d, d);
            RequireLength(_embedNorm.Shape.ElementCount, d, Ernie5WeightNames.MtpEmbedNorm);
            RequireLength(_hiddenNorm.Shape.ElementCount, d, Ernie5WeightNames.MtpHiddenNorm);
            _block = new Ernie5Layer(backend, cfg, index, source, hcaRope, dsaRope);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Layer index, from <c>num_hidden_layers</c> on.</summary>
    public int Index { get; }

    /// <summary>Runs the layer on <paramref name="tokens"/> rows: <paramref name="previous"/> is the target's (or an earlier draft step's)
    /// residual stream <c>[tokens, hc, hidden]</c>, <paramref name="embedded"/> the next tokens' embeddings <c>[tokens, hidden]</c>. Returns the
    /// layer's output stream, <c>[tokens, hc, hidden]</c>, which a later draft step takes as its <paramref name="previous"/>.</summary>
    public float[] Forward(ReadOnlySpan<float> previous, ReadOnlySpan<float> embedded, int tokens, int startPos, Ernie5SequenceState state)
    {
        int d = _cfg.HiddenSize, hc = _cfg.HcMult;
        float eps = (float)_cfg.RmsNormEps;
        float[] e = embedded.ToArray();
        HostOps.RmsNorm(_backend, e, d, _embedNorm, eps);
        float[] eProj = HostOps.Project(_backend, e, tokens, d, _embedProj, d);
        float[] h = previous.ToArray();
        HostOps.RmsNorm(_backend, h, d, _hiddenNorm, eps);
        float[] stream = HostOps.Project(_backend, h, tokens * hc, d, _hiddenProj, d);
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> row = eProj.AsSpan(t * d, d);
            for (int c = 0; c < hc; c++)
            {
                Span<float> target = stream.AsSpan((t * hc + c) * d, d);
                for (int i = 0; i < d; i++) target[i] += row[i];
            }
        }
        _block.Forward(stream, tokens, startPos, state);
        return stream;
    }

    private static Tensor ReadMatrix(IErnie5WeightSource source, string name, int rows, int cols)
    {
        Tensor t = source.Read(name);
        if (t.Shape.Rank == 2 && t.Shape[0] == rows && t.Shape[1] == cols) return t;
        string shape = t.Shape.ToString();
        t.Dispose();
        throw new InvalidDataException($"'{name}' is {shape}; the config needs [{rows}, {cols}].");
    }

    private static void RequireLength(long actual, long expected, string leaf)
    {
        if (actual != expected) throw new InvalidDataException($"MTP '{leaf}' holds {actual} values; the config needs {expected}.");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _block?.Dispose();
        _embedNorm?.Dispose();
        _hiddenNorm?.Dispose();
        _embedProj?.Dispose();
        _hiddenProj?.Dispose();
    }
}
