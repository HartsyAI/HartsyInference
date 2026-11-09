using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Query tokens from prompt 3D points: a learned Fourier embedding of the voxel coordinate plus a projection of the
/// latent feature at that voxel, then a LayerNorm. Borrows its weights.</summary>
internal sealed class CoLa3DPointPromptEncoder
{
    private const float NormEps = 1e-5f;

    private readonly int _width, _resolution, _fourierFeatures;
    private readonly Tensor _wr;
    private readonly Dense _fourierUp, _fourierDown, _latentProj;
    private readonly Affine _outputNorm;

    internal CoLa3DPointPromptEncoder(IReadOnlyDictionary<string, Tensor> w, CoLa3DConfig config)
    {
        _width = config.HiddenDim;
        _resolution = config.LatentResolution;
        _fourierFeatures = config.PointFourierFeatures;
        _wr = w["pointPrompt.fourierWr.weight"];
        _fourierUp = LoadDense(w, "pointPrompt.fourierUp");
        _fourierDown = LoadDense(w, "pointPrompt.fourierDown");
        _latentProj = LoadDense(w, "pointPrompt.latentProj");
        _outputNorm = LoadAffine(w, "pointPrompt.outputNorm");
    }

    /// <summary>Owned query rows <c>[1, B·Q, C]</c>. <paramref name="points"/> holds <c>B·Q·3</c> voxel coordinates and
    /// <paramref name="latentRows"/> the matching latent feature <c>[1, B·Q, D]</c>.</summary>
    internal Tensor Encode(IBackend backend, ReadOnlySpan<int> points, Tensor latentRows)
    {
        int n = (int)latentRows.Shape[1];
        Tensor fourier = Alloc(1, n, 2L * _fourierFeatures);
        FourierFeatures(points, fourier.AsSpan<float>());
        Tensor hidden = Linear(backend, fourier, _fourierUp, _width);
        fourier.Dispose();
        Tensor act = new(hidden.Shape, DType.F32);
        backend.GeluErf(act, hidden);
        hidden.Dispose();
        Tensor coord = Linear(backend, act, _fourierDown, _width);
        act.Dispose();

        Tensor latent = Linear(backend, latentRows, _latentProj, _width);
        Tensor sum = new(coord.Shape, DType.F32);
        backend.Add(sum, coord, latent);
        coord.Dispose();
        latent.Dispose();
        Tensor output = Norm(backend, sum, _outputNorm, NormEps);
        sum.Dispose();
        return output;
    }

    /// <summary>Gathers, for every sample and prompt, the row of <paramref name="source"/> at that prompt's voxel. Invalid
    /// prompts give zeros; a valid prompt with no voxel at its coordinate is an error.</summary>
    internal static Tensor GatherLatent(
        Tensor source,
        (int Start, int End)[] ranges,
        int[] coords,
        int[] points,
        bool[]? valid,
        int queries,
        int resolution)
    {
        int width = (int)source.Shape[source.Shape.Rank - 1];
        int total = ranges.Length * queries;
        Tensor output = Alloc(1, total, width);
        ReadOnlySpan<float> src = source.AsReadOnlySpan<float>();
        Span<float> dst = output.AsSpan<float>();
        for (int b = 0; b < ranges.Length; b++)
        {
            Dictionary<long, int> rowByVoxel = new();
            for (int r = ranges[b].Start; r < ranges[b].End; r++)
                rowByVoxel[VoxelIndex(coords[r * 4 + 1], coords[r * 4 + 2], coords[r * 4 + 3], resolution)] = r;

            for (int q = 0; q < queries; q++)
            {
                int slot = b * queries + q;
                if (valid is not null && !valid[slot])
                    continue;
                int px = points[slot * 3], py = points[slot * 3 + 1], pz = points[slot * 3 + 2];
                if ((uint)px >= (uint)resolution || (uint)py >= (uint)resolution || (uint)pz >= (uint)resolution)
                    throw new HartsyInferenceException($"Point prompt ({px}, {py}, {pz}) of sample {b} lies outside the {resolution}³ latent grid.");
                if (!rowByVoxel.TryGetValue(VoxelIndex(px, py, pz, resolution), out int row))
                    throw new HartsyInferenceException($"Point prompt ({px}, {py}, {pz}) of sample {b} has no active voxel in the latent.");
                src.Slice(row * width, width).CopyTo(dst.Slice(slot * width, width));
            }
        }

        return output;
    }

    /// <summary><c>[cos(2πh), sin(2πh)]</c> with <c>h = (coord / resolution) · Wrᵀ</c>, the <c>LearnableFourierPE</c> layout.</summary>
    private void FourierFeatures(ReadOnlySpan<int> points, Span<float> dst)
    {
        ReadOnlySpan<float> wr = _wr.AsReadOnlySpan<float>();
        int n = points.Length / 3;
        for (int i = 0; i < n; i++)
        {
            float x = points[i * 3] / (float)_resolution;
            float y = points[i * 3 + 1] / (float)_resolution;
            float z = points[i * 3 + 2] / (float)_resolution;
            Span<float> row = dst.Slice(i * 2 * _fourierFeatures, 2 * _fourierFeatures);
            for (int f = 0; f < _fourierFeatures; f++)
            {
                float h = wr[f * 3] * x + wr[f * 3 + 1] * y + wr[f * 3 + 2] * z;
                float angle = 2f * MathF.PI * h;
                row[f] = MathF.Cos(angle);
                row[_fourierFeatures + f] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>Linear index of a voxel in a grid of <paramref name="resolution"/> cells per axis.</summary>
    private static long VoxelIndex(int x, int y, int z, int resolution) => ((long)x * resolution + y) * resolution + z;
}
