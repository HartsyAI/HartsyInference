using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Stage 1 of UniRig: point encoder, projection into the decoder width, OPT decoder and skeleton token sampling, built from one skeleton checkpoint.</summary>
public sealed class UniRigSkeletonModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly UniRigShapeEncoder _encoder;
    private readonly UniRigDecoder _decoder;
    private readonly Tensor _projectionWeight;
    private readonly Tensor _projectionBias;
    private bool _disposed;

    /// <param name="checkpoint">The skeleton checkpoint; it must outlive this model.</param>
    public UniRigSkeletonModel(IBackend backend, UniRigCheckpoint checkpoint, UniRigShapeEncoderConfig? encoderConfig = null,
        UniRigDecoderConfig? decoderConfig = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        UniRigShapeEncoderConfig encoder = encoderConfig ?? UniRigShapeEncoderConfig.Default;
        UniRigDecoderConfig decoder = decoderConfig ?? UniRigDecoderConfig.Default;
        _backend = backend;
        _encoder = new UniRigShapeEncoder(backend, encoder, checkpoint.Tensors, "mesh_encoder.encoder");
        _projectionWeight = checkpoint.Get("output_proj.weight", decoder.HiddenSize, encoder.Width);
        _projectionBias = checkpoint.Get("output_proj.bias", decoder.HiddenSize);
        _decoder = new UniRigDecoder(backend, decoder, checkpoint.Tensors);
    }

    /// <summary>
    /// Samples a skeleton for a point cloud in the normalized cube. A draw that does not end in a decodable skeleton is retried
    /// with the next seed, up to <see cref="UniRigSamplingOptions.MaxAttempts"/> times.
    /// </summary>
    /// <param name="points">Positions, length <c>3 * N</c>.</param>
    /// <param name="normals">Normals, length <c>3 * N</c>.</param>
    public UniRigSkeleton Generate(ReadOnlySpan<float> points, ReadOnlySpan<float> normals, UniRigSamplingOptions options,
        CancellationToken cancel = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(options);
        HartsyInferenceException? last = null;
        for (int attempt = 0; attempt < Math.Max(1, options.MaxAttempts); attempt++)
        {
            cancel.ThrowIfCancellationRequested();
            UniRigSamplingOptions run = options with { Seed = options.Seed + attempt };
            try
            {
                return GenerateOnce(points, normals, run, cancel);
            }
            catch (HartsyInferenceException ex)
            {
                last = ex;
            }
        }

        string message = $"UniRig produced no decodable skeleton in {options.MaxAttempts} attempts: {last?.Message}";
        throw last is null ? new HartsyInferenceException(message) : new HartsyInferenceException(message, last);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _decoder.Dispose();
        _encoder.Dispose();
    }

    private UniRigSkeleton GenerateOnce(ReadOnlySpan<float> points, ReadOnlySpan<float> normals, UniRigSamplingOptions options,
        CancellationToken cancel)
    {
        using Tensor tokens = _encoder.Encode(points, normals, options.Seed);
        using Tensor condition = new(new TensorShape(1, tokens.Shape[1], _projectionBias.Shape[0]), DType.F32);
        _backend.Linear(condition, tokens, _projectionWeight, _projectionBias);
        int cls = UniRigTokenizer.ArticulationXlToken;
        float[] first = _decoder.Begin(condition, [UniRigTokenizer.BosToken, cls]);
        int[] sequence = UniRigTokenSampler.Run(first, _decoder.Step, options, cls, cancel);
        return UniRigTokenizer.Detokenize(sequence);
    }
}
