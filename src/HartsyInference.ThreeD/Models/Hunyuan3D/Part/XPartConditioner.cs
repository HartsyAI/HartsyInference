using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>One part's conditioning tokens and the object's, each <c>[Q, 1024]</c> as <see cref="Tensor"/> <c>[1, Q, 1024]</c>.</summary>
/// <param name="ObjectTokens">The object tokens (the same for every part); the caller disposes them.</param>
/// <param name="PartTokens">One <c>geo_cond</c> per part, in the order of the parts passed in; the caller disposes them.</param>
public sealed record XPartConditioning(Tensor ObjectTokens, IReadOnlyList<Tensor> PartTokens);

/// <summary>The surface samples of one part: the points of the object's surface that fall in its box, <c>[M, 3]</c> and their features <c>[M, 4]</c>, with the indices of the query points.</summary>
public sealed record XPartPartSample(float[] Points, float[] Features, int[] QueryIndices);

/// <summary>The X-Part conditioner (<c>Conditioner.forward</c> of <c>partgen/models/conditioner/condioner_release.py</c>, <c>use_geo</c>, <c>use_obj</c> and <c>use_seg_feat</c> on): the object
/// ShapeVAE encoder, the part-in-box ShapeVAE encoder and the Sonata feature extractor, each token of either encoder extended by the Sonata feature of the object point nearest to its query
/// point (<c>torch.cdist</c> argmin), and projected from 1536 to 1024 channels by <c>obj_out_proj</c> and <c>geo_out_proj</c>.</summary>
/// <remarks>Upstream's randomness is the caller's here: the query indices (upstream draws <c>fps</c> from a random start) and the voxel pick of the Sonata grid sample (upstream picks a random
/// point per voxel; <see cref="P3SamVoxelPick.First"/> is the reproducible one). Weights come from <c>conditioner.safetensors</c>; BF16 widens to F32.</remarks>
public sealed class XPartConditioner : IDisposable
{
    private const int TokenWidth = 1024, SonataWidth = 512;

    private readonly IBackend _backend;
    private readonly XPartShapeEncoder _objectEncoder, _partEncoder;
    private readonly P3SamFeatures _sonata;
    private readonly Tensor _objectProjW, _objectProjB, _partProjW, _partProjB;
    private readonly List<Tensor> _owned = [];
    private bool _disposed;

    /// <summary>Builds the conditioner from the tensors of <c>conditioner.safetensors</c>; the caller keeps ownership of <paramref name="weights"/>.</summary>
    public XPartConditioner(IBackend backend, IReadOnlyDictionary<string, Tensor> weights, XPartShapeEncoderConfig? encoderConfig = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        _backend = backend;
        _objectEncoder = new XPartShapeEncoder(backend, encoderConfig);
        _partEncoder = new XPartShapeEncoder(backend, encoderConfig);
        try
        {
            _objectEncoder.LoadWeights(weights, "obj_encoder");
            _partEncoder.LoadWeights(weights, "geo_encoder.local_encoder");
            const string Prefix = "seg_feat_encoder.";
            Dictionary<string, Tensor> sonata = [];
            foreach ((string key, Tensor value) in weights)
            {
                if (key.StartsWith(Prefix, StringComparison.Ordinal))
                    sonata[key[Prefix.Length..]] = value;
            }

            _sonata = new P3SamFeatures(backend, sonata);
            _objectProjW = Widen(weights, "obj_out_proj.weight", TokenWidth, TokenWidth + SonataWidth);
            _objectProjB = Widen(weights, "obj_out_proj.bias", TokenWidth);
            _partProjW = Widen(weights, "geo_out_proj.weight", TokenWidth, TokenWidth + SonataWidth);
            _partProjB = Widen(weights, "geo_out_proj.bias", TokenWidth);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The Sonata feature of every object point, <c>[N, 512]</c>, row-major. The object points are the caller's surface samples (upstream: 81,920 of them, in the 0.8 box).</summary>
    public float[] SonataFeatures(ReadOnlySpan<float> points, ReadOnlySpan<float> features, P3SamVoxelPick pick = P3SamVoxelPick.First)
    {
        int count = ObjectCount(points, features);
        double[] xyz = new double[3 * count];
        float[] normals = new float[3 * count];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                xyz[3 * i + c] = points[3 * i + c];
                normals[3 * i + c] = features[4 * i + c];
            }
        }

        return _sonata.Extract(xyz, normals, count, pick);
    }

    /// <summary>Runs the conditioner. <paramref name="points"/> <c>[N, 3]</c> and <paramref name="features"/> <c>[N, 4]</c> (normal, sharp-edge label) are the object's surface samples,
    /// <paramref name="objectQueries"/> the indices of the object encoder's query points; each entry of <paramref name="parts"/> is one part's samples with its own query indices.</summary>
    public XPartConditioning Condition(ReadOnlySpan<float> points, ReadOnlySpan<float> features, ReadOnlySpan<int> objectQueries, IReadOnlyList<XPartPartSample> parts,
        P3SamVoxelPick pick = P3SamVoxelPick.First)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(parts);
        ObjectCount(points, features);
        float[] sonata = SonataFeatures(points, features, pick);

        Tensor objectTokens = Project(_objectEncoder.Encode(points, features, objectQueries), sonata, Nearest(points, objectQueries, points), _objectProjW, _objectProjB);
        List<Tensor> partTokens = [];
        try
        {
            foreach (XPartPartSample part in parts)
            {
                if (part.Points.Length % 3 != 0 || part.Features.Length != part.Points.Length / 3 * 4)
                    throw new ArgumentException("A part needs 3 floats and 4 features per point.", nameof(parts));
                Tensor encoded = _partEncoder.Encode(part.Points, part.Features, part.QueryIndices);
                partTokens.Add(Project(encoded, sonata, Nearest(part.Points, part.QueryIndices, points), _partProjW, _partProjB));
            }
        }
        catch
        {
            objectTokens.Dispose();
            foreach (Tensor t in partTokens)
                t.Dispose();
            throw;
        }

        return new XPartConditioning(objectTokens, partTokens);
    }

    /// <summary>Releases the encoders and the converted weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _objectEncoder?.Dispose();
        _partEncoder?.Dispose();
        _sonata?.Dispose();
        foreach (Tensor t in _owned)
            t.Dispose();
    }

    // concat([tokens, sonata[nearest]]) then the output projection: [1, Q, 1024] in, [1, Q, 1024] out (consumes tokens).
    private Tensor Project(Tensor tokens, float[] sonata, int[] nearest, Tensor weight, Tensor bias)
    {
        using (tokens)
        {
            int q = nearest.Length;
            using Tensor joined = new(new TensorShape(1, q, TokenWidth + SonataWidth), DType.F32);
            Span<float> target = joined.AsSpan<float>();
            ReadOnlySpan<float> source = tokens.AsSpan<float>();
            for (int i = 0; i < q; i++)
            {
                source.Slice(i * TokenWidth, TokenWidth).CopyTo(target.Slice(i * (TokenWidth + SonataWidth), TokenWidth));
                sonata.AsSpan(nearest[i] * SonataWidth, SonataWidth).CopyTo(target.Slice(i * (TokenWidth + SonataWidth) + TokenWidth, SonataWidth));
            }

            Tensor projected = new(new TensorShape(1, q, TokenWidth), DType.F32);
            _backend.Linear(projected, joined, weight, bias);
            return projected;
        }
    }

    // For each query point (a row of queryPoints picked by queryIndices), the index of the nearest object point: torch.cdist(...).argmin with the first minimum on ties.
    private static int[] Nearest(ReadOnlySpan<float> queryPoints, ReadOnlySpan<int> queryIndices, ReadOnlySpan<float> objectPoints)
    {
        float[] queries = new float[3 * queryIndices.Length];
        for (int i = 0; i < queryIndices.Length; i++)
            queryPoints.Slice(3 * queryIndices[i], 3).CopyTo(queries.AsSpan(3 * i, 3));
        float[] objects = objectPoints.ToArray();
        int objectCount = objects.Length / 3;
        int[] nearest = new int[queryIndices.Length];
        Parallel.For(0, nearest.Length, i =>
        {
            float qx = queries[3 * i], qy = queries[3 * i + 1], qz = queries[3 * i + 2], best = float.MaxValue;
            int arg = 0;
            for (int j = 0; j < objectCount; j++)
            {
                float dx = objects[3 * j] - qx, dy = objects[3 * j + 1] - qy, dz = objects[3 * j + 2] - qz;
                float d = dx * dx + dy * dy + dz * dz;
                if (d < best)
                {
                    best = d;
                    arg = j;
                }
            }

            nearest[i] = arg;
        });
        return nearest;
    }

    private static int ObjectCount(ReadOnlySpan<float> points, ReadOnlySpan<float> features)
    {
        int count = points.Length / 3;
        if (count == 0 || points.Length != 3 * count || features.Length != 4 * count)
            throw new ArgumentException($"The object needs 3 floats and 4 features per point; got {points.Length} and {features.Length}.");
        return count;
    }

    private Tensor Widen(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new KeyNotFoundException($"The X-Part conditioner tensor '{key}' is missing.");
        if (!source.Shape.Equals(new TensorShape(shape)))
            throw new InvalidDataException($"The X-Part conditioner tensor '{key}' has shape {source.Shape}, expected {new TensorShape(shape)}.");
        Tensor widened = Hunyuan3DShapeVae.F32(source);
        if (ReferenceEquals(widened, source))
            return source;
        _owned.Add(widened);
        return widened;
    }
}
