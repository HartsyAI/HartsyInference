using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The geometry of an <see cref="LtxVideo2PackedTextConnector"/>; the defaults are LTX-2.0's
/// (<c>connectors/config.json</c>).</summary>
public sealed record LtxVideo2PackedTextConnectorConfig
{
    /// <summary>Width of one text-tower hidden state, and of the connector.</summary>
    public int CaptionChannels { get; init; } = 3840;

    /// <summary>Hidden states stacked per token: the tower's embedding output plus one per layer (<c>text_proj_in_factor</c>).</summary>
    public int States { get; init; } = 49;

    /// <summary>Attention heads of the connector.</summary>
    public int Heads { get; init; } = 30;

    /// <summary>Width of one attention head.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Connector blocks.</summary>
    public int Layers { get; init; } = 2;

    /// <summary>Learnable registers that replace the padding; the sequence length is a multiple of this.</summary>
    public int Registers { get; init; } = 128;

    /// <summary>The sequence length the connector's 1-D rotation is normalized by.</summary>
    public int RopeBaseSeqLen { get; init; } = 4096;

    /// <summary>The rotation's frequency base.</summary>
    public float RopeTheta { get; init; } = 10000f;

    /// <summary>The rotation's pairing.</summary>
    public LtxVideo2Rope.RopeType RopeType { get; init; } = LtxVideo2Rope.RopeType.Split;

    /// <summary>Epsilon of the connector's norms and its QK norm.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>The factor the normalized hidden states are scaled by before the projection.</summary>
    public float PackScale { get; init; } = 8f;
}

/// <summary>LTX-2.0's text conditioning after the text tower (diffusers <c>LTX2Pipeline._pack_text_embeds</c> and the video
/// branch of <c>LTX2TextConnectors</c>): every stacked hidden state is centred on its mean and scaled by its range over the
/// prompt's tokens, the stack is projected to one vector per token (<c>text_proj_in</c>), and the connector replaces the
/// padding with learnable registers and runs its self-attention blocks. The result is the <c>[L, captionChannels]</c>
/// prompt embedding the transformer's own caption projection takes; every position attends, so no mask comes with it.</summary>
/// <remarks>LTX-2.3 normalizes per token and projects per modality instead (<see cref="LtxVideo2TextConnectors"/>); the
/// connector blocks are the same and are shared. Only <c>text_proj_in.*</c> and <c>video_connector.*</c> are read.
/// Upstream left-pads the prompt to the sequence length and masks the padding in the tower; encoding the prompt's own
/// tokens alone gives the same hidden states (the tower's rotation is relative), so this takes the unpadded states and the
/// sequence length separately and never projects padding rows.</remarks>
public sealed unsafe class LtxVideo2PackedTextConnector : IDisposable
{
    private const float PackEps = 1e-6f;
    private const string ProjectionKey = "text_proj_in.weight";
    private const string ConnectorPrefix = "video_connector";

    private readonly LtxVideo2PackedTextConnectorConfig _config;
    private readonly LtxVideo2TextConnectors.Connector _connector;
    private Tensor? _projection;

    /// <summary>Creates the connector for <paramref name="config"/>.</summary>
    public LtxVideo2PackedTextConnector(LtxVideo2PackedTextConnectorConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Heads * config.HeadDim != config.CaptionChannels)
            throw new ArgumentException($"The connector is {config.Heads}×{config.HeadDim} wide but the projection emits "
                + $"{config.CaptionChannels}.", nameof(config));
        _config = config;
        _connector = new LtxVideo2TextConnectors.Connector(config.CaptionChannels, config.Heads, config.HeadDim, config.NormEps,
            config.NormEps, config.Layers, config.Registers,
            LtxVideo2Rope.ForConnector1d(config.CaptionChannels, config.RopeTheta, config.RopeBaseSeqLen, config.RopeType,
                config.Heads, config.HeadDim));
    }

    /// <summary>The configuration the connector was built for.</summary>
    public LtxVideo2PackedTextConnectorConfig Config => _config;

    /// <summary>Whether <paramref name="key"/> of a diffusers <c>LTX2TextConnectors</c> checkpoint is read here.</summary>
    public static bool IsVideoKey(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return key.StartsWith("text_proj_in.", StringComparison.Ordinal) || key.StartsWith(ConnectorPrefix + ".", StringComparison.Ordinal);
    }

    /// <summary>Binds the checkpoint's tensors, which stay the caller's.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        Tensor projection = w[ProjectionKey];
        long features = (long)_config.CaptionChannels * _config.States;
        if (projection.Shape.Rank != 2 || projection.Shape[0] != _config.CaptionChannels || projection.Shape[1] != features)
            throw new InvalidOperationException($"'{ProjectionKey}' is {projection.Shape}; expected [{_config.CaptionChannels}, {features}].");
        _projection = projection;
        _connector.LoadWeights(w, ConnectorPrefix, blocks: "transformer_blocks");
    }

    /// <summary>Every weight the forward reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projection is not null)
            yield return _projection;
        foreach (Tensor t in _connector.EnumerateWeights())
            yield return t;
    }

    /// <summary>The prompt embedding for one prompt.</summary>
    /// <param name="hiddenStates">The text tower's stacked hidden states for the prompt's own tokens, F32
    /// <c>[1, tokens, States·CaptionChannels]</c> state-major (state 0 the embedding output), borrowed.</param>
    /// <param name="sequenceLength">The length the prompt is padded to with registers; a multiple of the register count
    /// (upstream pads to 1024).</param>
    /// <returns>Host F32 <c>[sequenceLength, CaptionChannels]</c>; the caller owns it.</returns>
    public Tensor Forward(IBackend backend, Tensor hiddenStates, int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hiddenStates);
        Tensor projection = _projection ?? throw new InvalidOperationException("LtxVideo2PackedTextConnector.LoadWeights has not run.");
        int channels = _config.CaptionChannels, states = _config.States;
        if (hiddenStates.DType != DType.F32 || hiddenStates.Shape.Rank != 3 || hiddenStates.Shape[0] != 1
            || hiddenStates.Shape[2] != (long)states * channels)
            throw new ArgumentException($"The connector takes F32 hidden states [1, tokens, {(long)states * channels}]; got "
                + $"{hiddenStates.DType} {hiddenStates.Shape}.", nameof(hiddenStates));
        int tokens = (int)hiddenStates.Shape[1];
        if (sequenceLength < 1 || sequenceLength % _config.Registers != 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), sequenceLength,
                $"The sequence length must be a positive multiple of the {_config.Registers} registers.");
        if (tokens < 1 || tokens > sequenceLength)
            throw new ArgumentException($"The prompt has {tokens} tokens for a sequence of {sequenceLength}.", nameof(hiddenStates));

        using Tensor packed = Pack(hiddenStates, tokens);
        using Tensor projected = new(new TensorShape(tokens, channels), DType.F32);
        backend.Linear(projected, packed, projection, null);

        // The prompt's rows lead; the connector fills every later position with its registers.
        using Tensor features = new(new TensorShape(sequenceLength, channels), DType.F32);
        new Span<float>((void*)features.DataPointer, checked(sequenceLength * channels)).Clear();
        Buffer.MemoryCopy((void*)projected.DataPointer, (void*)features.DataPointer, (long)sequenceLength * channels * sizeof(float),
            (long)tokens * channels * sizeof(float));
        float[] valid = new float[sequenceLength];
        valid.AsSpan(0, tokens).Fill(1f);
        using Tensor connected = _connector.Forward(backend, features, valid);
        Tensor host = new(connected.Shape, DType.F32, DeviceKind.Cpu);
        try
        {
            backend.CopyTo(host, connected);
            return host;
        }
        catch
        {
            host.Dispose();
            throw;
        }
    }

    // Per stacked state: (x − mean) / (max − min + eps) · scale over the prompt's tokens and channels, relaid channel-major
    // (feature = channel·States + state), which is the layout text_proj_in was trained on.
    private Tensor Pack(Tensor hiddenStates, int tokens)
    {
        int channels = _config.CaptionChannels, states = _config.States;
        long width = (long)states * channels;
        Tensor packed = new(new TensorShape(tokens, width), DType.F32);
        float* source = (float*)hiddenStates.DataPointer;
        float* destination = (float*)packed.DataPointer;
        float valid = (float)tokens * channels + PackEps;
        for (int state = 0; state < states; state++)
        {
            double sum = 0.0;
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            for (int t = 0; t < tokens; t++)
            {
                float* row = source + t * width + (long)state * channels;
                for (int c = 0; c < channels; c++)
                {
                    float value = row[c];
                    sum += value;
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
            }

            float mean = (float)(sum / valid);
            float range = max - min + PackEps;
            for (int t = 0; t < tokens; t++)
            {
                float* row = source + t * width + (long)state * channels;
                float* target = destination + t * width + state;
                for (int c = 0; c < channels; c++)
                    target[(long)c * states] = (row[c] - mean) / range * _config.PackScale;
            }
        }

        return packed;
    }

    /// <inheritdoc />
    public void Dispose() => _connector.Dispose();
}
