using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Stage 2 of UniRig: per-point skin weights from a point transformer over the mesh, a Michelangelo-style global encoder, a bone
/// encoder, mesh-to-bone cross-attention, per-head bone scoring and a scoring MLP that also reads the geometric voxel prior.
/// Built from one skin checkpoint; inference only.
/// </summary>
public sealed class UniRigSkinModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly UniRigSkinConfig _config;
    private readonly UniRigPtv3Encoder _pointEncoder;
    private readonly UniRigShapeEncoder _globalEncoder;
    private readonly UniRigResidualCrossAttn[] _boneAttn;
    private readonly UniRigResidualCrossAttn[] _meshAttn;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private bool _disposed;

    /// <param name="checkpoint">The skin checkpoint; it must outlive this model.</param>
    public UniRigSkinModel(IBackend backend, UniRigCheckpoint checkpoint, UniRigSkinConfig? config = null, UniRigPtv3Config? pointConfig = null,
        UniRigShapeEncoderConfig? globalConfig = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        _backend = backend;
        _config = config ?? new UniRigSkinConfig();
        _w = checkpoint.Tensors;
        _pointEncoder = new UniRigPtv3Encoder(backend, pointConfig ?? UniRigPtv3Config.Default, _w, "mesh_encoder");
        _globalEncoder = new UniRigShapeEncoder(backend, globalConfig ?? UniRigShapeEncoderConfig.Default, _w, "global_encoder.encoder");
        _boneAttn = new UniRigResidualCrossAttn[_config.NumBoneAttn];
        for (int i = 0; i < _boneAttn.Length; i++)
        {
            _boneAttn[i] = new UniRigResidualCrossAttn(backend, _w, $"bone_encoder.attn.{i}", _config.FeatDim, _config.NumHeads);
        }

        _meshAttn = new UniRigResidualCrossAttn[_config.NumMeshBoneAttn];
        for (int i = 0; i < _meshAttn.Length; i++)
        {
            _meshAttn[i] = new UniRigResidualCrossAttn(backend, _w, $"mesh_bone_attn.{i}", _config.FeatDim, _config.NumHeads);
        }
    }

    /// <summary>
    /// Skin weights <c>[N, J]</c> for sampled points, each row summing to 1.
    /// </summary>
    /// <param name="points">Sampled positions, length <c>3 * N</c> (the encoders' coordinate frame).</param>
    /// <param name="normals">Sampled normals, length <c>3 * N</c>.</param>
    /// <param name="joints">Joint positions in the same frame, length <c>3 * J</c>.</param>
    /// <param name="parents">Parent per joint, <c>-1</c> for roots; parents precede children.</param>
    /// <param name="voxelSkin">Geometric prior per sampled point <c>[N, J]</c>.</param>
    /// <param name="seed">Seed of the global encoder's point subset.</param>
    public float[] Predict(float[] points, float[] normals, float[] joints, int[] parents, float[] voxelSkin, int seed, CancellationToken cancel = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int n = points.Length / 3;
        int j = parents.Length;
        if (n == 0 || j == 0 || normals.Length != 3 * n || joints.Length != 3 * j || voxelSkin.Length != n * j)
        {
            throw new ArgumentException($"UniRig skin inputs disagree: {n} points, {j} joints, {voxelSkin.Length} prior values.");
        }

        for (int i = 0; i < j; i++)
        {
            if (parents[i] >= i)
            {
                throw new ArgumentException($"UniRig joint {i} has parent {parents[i]}; parents must precede children.", nameof(parents));
            }
        }

        int d = _config.FeatDim;
        float[] meshFeatures = MeshFeatures(points, normals, d);
        cancel.ThrowIfCancellationRequested();
        using Tensor globalLatents = GlobalLatents(points, normals, seed);
        using Tensor bone0 = BoneEmbedding(points, joints, j);
        using Tensor boneContext = Concat(bone0, globalLatents);
        Tensor bone = bone0;
        bool ownsBone = false;
        foreach (UniRigResidualCrossAttn attn in _boneAttn)
        {
            Tensor next = attn.Forward(bone, boneContext);
            if (ownsBone)
            {
                bone.Dispose();
            }

            bone = next;
            ownsBone = true;
        }

        try
        {
            using Tensor context = Concat(bone, globalLatents);
            float[] attended = MeshBoneAttention(meshFeatures, context, n, d, cancel);
            return Score(attended, bone, voxelSkin, parents, n, j, cancel);
        }
        finally
        {
            if (ownsBone)
            {
                bone.Dispose();
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _pointEncoder.Dispose();
        _globalEncoder.Dispose();
        foreach (UniRigResidualCrossAttn block in _boneAttn)
        {
            block.Dispose();
        }

        foreach (UniRigResidualCrossAttn block in _meshAttn)
        {
            block.Dispose();
        }
    }

    private Tensor W(string name, params long[] shape) => UniRigCheckpoint.Get(_w, name, shape);

    private float[] MeshFeatures(float[] points, float[] normals, int d)
    {
        using Tensor raw = _pointEncoder.Encode(points, normals);
        using Tensor projected = UniRigOps.Linear(_backend, raw, W("feat_map.0.weight", d, raw.Shape[1]), W("feat_map.0.bias", d));
        using Tensor normed = UniRigOps.LayerNorm(_backend, projected, W("feat_map.1.weight", d), W("feat_map.1.bias", d));
        using Tensor activated = UniRigOps.GeluErf(_backend, normed);
        return activated.AsReadOnlySpan<float>().ToArray();
    }

    private Tensor GlobalLatents(float[] points, float[] normals, int seed)
    {
        int d = _config.FeatDim;
        using Tensor tokens = _globalEncoder.Encode(points, normals, seed);
        using Tensor projected = UniRigOps.Linear(_backend, tokens, W("out_proj.0.weight", d, _globalEncoder.Width), W("out_proj.0.bias", d));
        using Tensor normed = UniRigOps.LayerNorm(_backend, projected, W("out_proj.1.weight", d), W("out_proj.1.bias", d));
        return UniRigOps.GeluErf(_backend, normed);
    }

    private Tensor BoneEmbedding(float[] points, float[] joints, int j)
    {
        int freqs = _config.BoneFreqs;
        int width = 3 * (2 * freqs + 1);
        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue];
        for (int i = 0; i < points.Length; i++)
        {
            min[i % 3] = Math.Min(min[i % 3], points[i]);
        }

        float[] rows = new float[j * width];
        for (int b = 0; b < j; b++)
        {
            int at = b * width;
            for (int c = 0; c < 3; c++)
            {
                rows[at + c] = joints[3 * b + c] - min[c];
            }

            for (int c = 0; c < 3; c++)
            {
                for (int f = 0; f < freqs; f++)
                {
                    double angle = rows[at + c] * Math.Pow(2, f) * Math.PI;
                    rows[at + 3 + c * freqs + f] = (float)Math.Sin(angle);
                    rows[at + 3 + 3 * freqs + c * freqs + f] = (float)Math.Cos(angle);
                }
            }
        }

        int e = _config.BoneEmbedDim;
        int d = _config.FeatDim;
        Tensor x = UniRigOps.FromFloats(rows, new TensorShape(1, j, width));
        x = Stage(x, "bone_encoder.bone_encoder.1", "bone_encoder.bone_encoder.2", e, width);
        x = Stage(x, "bone_encoder.bone_encoder.4", "bone_encoder.bone_encoder.5", 4 * e, e);
        return Stage(x, "bone_encoder.bone_encoder.7", "bone_encoder.bone_encoder.8", d, 4 * e);
    }

    /// <summary>Linear, LayerNorm, GELU; releases <paramref name="input"/>.</summary>
    private Tensor Stage(Tensor input, string linear, string norm, long outDim, long inDim)
    {
        using Tensor owned = input;
        using Tensor projected = UniRigOps.Linear(_backend, owned, W($"{linear}.weight", outDim, inDim), W($"{linear}.bias", outDim));
        using Tensor normed = UniRigOps.LayerNorm(_backend, projected, W($"{norm}.weight", outDim), W($"{norm}.bias", outDim));
        return UniRigOps.GeluErf(_backend, normed);
    }

    private Tensor Concat(Tensor first, Tensor second)
    {
        Tensor output = new(new TensorShape(1, first.Shape[1] + second.Shape[1], first.Shape[2]), DType.F32);
        _backend.Concat(output, [first, second], 1);
        return output;
    }

    private float[] MeshBoneAttention(float[] meshFeatures, Tensor context, int n, int d, CancellationToken cancel)
    {
        float[] result = new float[meshFeatures.Length];
        int chunk = _config.AttentionChunk;
        for (int start = 0; start < n; start += chunk)
        {
            cancel.ThrowIfCancellationRequested();
            int count = Math.Min(chunk, n - start);
            Tensor x = UniRigOps.FromFloats(meshFeatures.AsSpan(start * d, count * d), new TensorShape(1, count, d));
            foreach (UniRigResidualCrossAttn block in _meshAttn)
            {
                Tensor next = block.Forward(x, context);
                x.Dispose();
                x = next;
            }

            x.AsReadOnlySpan<float>().CopyTo(result.AsSpan(start * d, count * d));
            x.Dispose();
        }

        return result;
    }

    private float[] Score(float[] meshFeatures, Tensor bone, float[] voxelSkin, int[] parents, int n, int j, CancellationToken cancel)
    {
        int d = _config.FeatDim;
        int heads = _config.NumHeads;
        float[] boneKeys;
        using (Tensor keys = UniRigOps.Linear(_backend, bone, W("kmesh.weight", (long)d * heads, d), W("kmesh.bias", (long)d * heads)))
        {
            boneKeys = keys.AsReadOnlySpan<float>().ToArray();
        }

        Tensor[] headKeys = new Tensor[heads];
        for (int h = 0; h < heads; h++)
        {
            float[] slice = new float[j * d];
            for (int b = 0; b < j; b++)
            {
                Array.Copy(boneKeys, b * d * heads + h * d, slice, b * d, d);
            }

            headKeys[h] = UniRigOps.FromFloats(slice, new TensorShape(j, d));
        }

        float[] mask = (float[])voxelSkin.Clone();
        for (int i = 0; i < j; i++)
        {
            if (parents[i] >= 0)
            {
                for (int p = 0; p < n; p++)
                {
                    mask[p * j + parents[i]] += mask[p * j + i];
                }
            }
        }

        float[] skin = new float[n * j];
        try
        {
            for (int start = 0; start < n; start += _config.PointChunk)
            {
                cancel.ThrowIfCancellationRequested();
                int count = Math.Min(_config.PointChunk, n - start);
                float[] prediction = ScoreChunk(meshFeatures, headKeys, voxelSkin, start, count, j);
                for (int p = 0; p < count; p++)
                {
                    int row = (start + p) * j;
                    double total = 0;
                    for (int b = 0; b < j; b++)
                    {
                        double v = prediction[p * j + b] * Math.Pow(mask[row + b], _config.VoxelMask);
                        skin[row + b] = (float)v;
                        total += v;
                    }

                    for (int b = 0; b < j; b++)
                    {
                        skin[row + b] = total > 0 ? (float)(skin[row + b] / total) : prediction[p * j + b];
                    }
                }
            }
        }
        finally
        {
            foreach (Tensor key in headKeys)
            {
                key.Dispose();
            }
        }

        return skin;
    }

    /// <summary>Scores <paramref name="count"/> points against every bone: softmax over bones of the scoring MLP's output.</summary>
    private float[] ScoreChunk(float[] meshFeatures, Tensor[] headKeys, float[] voxelSkin, int start, int count, int j)
    {
        int d = _config.FeatDim;
        int heads = _config.NumHeads;
        int rows = count * j;
        using Tensor features = UniRigOps.FromFloats(meshFeatures.AsSpan(start * d, count * d), new TensorShape(count, d));
        using Tensor queries = UniRigOps.Linear(_backend, features, W("qmesh.weight", (long)d * heads, d), W("qmesh.bias", (long)d * heads));
        ReadOnlySpan<float> q = queries.AsReadOnlySpan<float>();
        float[] attention = new float[rows * heads];
        float scale = (float)(1.0 / Math.Sqrt(d));
        for (int h = 0; h < heads; h++)
        {
            float[] slice = new float[count * d];
            for (int p = 0; p < count; p++)
            {
                q.Slice(p * d * heads + h * d, d).CopyTo(slice.AsSpan(p * d, d));
            }

            using Tensor headQuery = UniRigOps.FromFloats(slice, new TensorShape(count, d));
            using Tensor scores = UniRigOps.Linear(_backend, headQuery, headKeys[h], null);
            ReadOnlySpan<float> s = scores.AsReadOnlySpan<float>();
            for (int p = 0; p < count; p++)
            {
                SoftmaxInto(s.Slice(p * j, j), scale, attention, p * j, heads, h);
            }
        }

        using Tensor attentionTensor = UniRigOps.FromFloats(attention, new TensorShape(rows, heads));
        using Tensor attentionNormed = UniRigOps.LayerNorm(_backend, attentionTensor, W("attn_skin_norm.weight", heads), W("attn_skin_norm.bias", heads));
        using Tensor prior = UniRigOps.FromFloats(voxelSkin.AsSpan(start * j, rows), new TensorShape(rows, 1));
        using Tensor priorEmbedded = UniRigOps.Linear(_backend, prior, W("voxel_skin_embed.weight", heads, 1), W("voxel_skin_embed.bias", heads));
        using Tensor priorNormed = UniRigOps.LayerNorm(_backend, priorEmbedded, W("voxel_skin_norm.weight", heads), W("voxel_skin_norm.bias", heads));
        float[] joined = new float[rows * 2 * heads];
        ReadOnlySpan<float> a = attentionNormed.AsReadOnlySpan<float>();
        ReadOnlySpan<float> v = priorNormed.AsReadOnlySpan<float>();
        for (int r = 0; r < rows; r++)
        {
            a.Slice(r * heads, heads).CopyTo(joined.AsSpan(r * 2 * heads, heads));
            v.Slice(r * heads, heads).CopyTo(joined.AsSpan(r * 2 * heads + heads, heads));
        }

        using Tensor joinedTensor = UniRigOps.FromFloats(joined, new TensorShape(rows, 2 * heads));
        using Tensor reduced = UniRigOps.Linear(_backend, joinedTensor, W("downscale.0.weight", heads, 2 * heads), W("downscale.0.bias", heads));
        using Tensor reducedNormed = UniRigOps.LayerNorm(_backend, reduced, W("downscale.1.weight", heads), W("downscale.1.bias", heads));
        Tensor x = UniRigOps.GeluErf(_backend, reducedNormed);
        int m = _config.MlpDim;
        x = HiddenLayer(x, 0, 1, m, heads);
        x = HiddenLayer(x, 3, 4, m, m);
        x = HiddenLayer(x, 6, 7, m, m);
        x = HiddenLayer(x, 9, 10, m, m);
        using Tensor output = x;
        using Tensor logits = UniRigOps.Linear(_backend, output, W("skinweight_pred.net.12.weight", 1, m), W("skinweight_pred.net.12.bias", 1));
        ReadOnlySpan<float> l = logits.AsReadOnlySpan<float>();
        float[] prediction = new float[rows];
        for (int p = 0; p < count; p++)
        {
            SoftmaxInto(l.Slice(p * j, j), 1f, prediction, p * j, 1, 0);
        }

        return prediction;
    }

    /// <summary>Linear, LayerNorm, GELU of the scoring MLP; releases <paramref name="input"/>.</summary>
    private Tensor HiddenLayer(Tensor input, int linear, int norm, long outDim, long inDim)
    {
        using Tensor owned = input;
        using Tensor projected = UniRigOps.Linear(_backend, owned, W($"skinweight_pred.net.{linear}.weight", outDim, inDim), W($"skinweight_pred.net.{linear}.bias", outDim));
        using Tensor normed = UniRigOps.LayerNorm(_backend, projected, W($"skinweight_pred.net.{norm}.weight", outDim), W($"skinweight_pred.net.{norm}.bias", outDim));
        return UniRigOps.GeluErf(_backend, normed);
    }

    /// <summary>Softmax over <paramref name="scores"/> after scaling; writes element <c>b</c> to <c>target[(row + b) * stride + offset]</c>.</summary>
    private static void SoftmaxInto(ReadOnlySpan<float> scores, float scale, float[] target, int row, int stride, int offset)
    {
        float max = float.NegativeInfinity;
        foreach (float value in scores)
        {
            max = Math.Max(max, value * scale);
        }

        double total = 0;
        for (int b = 0; b < scores.Length; b++)
        {
            total += Math.Exp(scores[b] * scale - max);
        }

        for (int b = 0; b < scores.Length; b++)
        {
            target[(row + b) * stride + offset] = (float)(Math.Exp(scores[b] * scale - max) / total);
        }
    }
}
