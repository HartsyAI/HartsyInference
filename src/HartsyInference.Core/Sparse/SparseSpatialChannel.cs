using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Sparse;

/// <summary>
/// Spatial-to-channel and channel-to-spatial resampling of sparse voxels by an integer factor: a coarse voxel owns
/// <c>factor³</c> children, ordered <c>(x%f) + f·(y%f) + f²·(z%f)</c>, and its features are the children's features
/// concatenated along the channel axis. Planning (<see cref="PlanSpatialToChannel"/>, <see cref="PlanChannelToSpatial"/>)
/// is host code; <see cref="PlanChannelToSpatial"/> also reads its logits from host memory. The feature moves
/// (<see cref="SpatialToChannelFeats"/>, <see cref="ChannelToSpatialFeats"/>) run through <see cref="IBackend"/> row
/// scatter/gather. <see cref="MeanChannelGroups"/> is CPU-only: it reads and writes host memory directly.
/// </summary>
public static unsafe class SparseSpatialChannel
{
    /// <summary>Plans a space-to-channel merge: input voxels sharing a coarse cell collapse into one output voxel.</summary>
    /// <remarks>Host-only: works on the coords array alone, no tensor access.</remarks>
    public static Plan PlanSpatialToChannel(int[] coords, int resolution, int factor)
    {
        ValidateGeometry(coords, resolution, factor);
        int n = coords.Length / 4, child = ChildCount(factor), extent = (resolution + factor - 1) / factor;
        long[] cells = new long[n];
        int[] sub = new int[n];
        for (int i = 0; i < n; i++)
        {
            int x = coords[i * 4 + 1], y = coords[i * 4 + 2], z = coords[i * 4 + 3];
            cells[i] = CellKey(coords[i * 4], x / factor, y / factor, z / factor, extent);
            sub[i] = x % factor + factor * (y % factor) + factor * factor * (z % factor);
        }

        long[] unique = (long[])cells.Clone();
        Array.Sort(unique);
        int m = 0;
        for (int i = 0; i < n; i++)
            if (m == 0 || unique[i] != unique[m - 1]) unique[m++] = unique[i];

        int[] outCoords = new int[m * 4];
        for (int r = 0; r < m; r++) DecodeCell(unique[r], extent, outCoords, r * 4);
        int[] index = new int[n];
        for (int i = 0; i < n; i++) index[i] = Array.BinarySearch(unique, 0, m, cells[i]) * child + sub[i];
        return new Plan(outCoords, index, extent, child);
    }

    /// <summary>Plans a channel-to-spatial expansion: each parent yields the children whose logit is positive, in child order.</summary>
    /// <param name="childLogits">F32 values <c>[N·factor³]</c>, one per parent child slot.</param>
    /// <remarks>CPU-only: reads <paramref name="childLogits"/> through <c>DataPointer</c>, so it must be host-resident.</remarks>
    public static Plan PlanChannelToSpatial(int[] coords, int resolution, int factor, Tensor childLogits)
    {
        ValidateGeometry(coords, resolution, factor);
        RequireF32(childLogits, nameof(childLogits));
        int n = coords.Length / 4, child = ChildCount(factor);
        long slots = (long)n * child;
        if (childLogits.ElementCount != slots)
        {
            throw new ArgumentException(
                $"Child logits must hold {slots} values ({n} parents x {child} children), got {childLogits.ElementCount}.",
                nameof(childLogits));
        }

        float* logits = (float*)childLogits.DataPointer;
        int count = 0;
        for (long i = 0; i < slots; i++) if (logits[i] > 0f) count++;

        int[] outCoords = new int[count * 4], index = new int[count];
        int o = 0;
        for (int i = 0; i < n; i++)
            for (int s = 0; s < child; s++)
            {
                if (!(logits[(long)i * child + s] > 0f)) continue;
                outCoords[o * 4] = coords[i * 4];
                outCoords[o * 4 + 1] = coords[i * 4 + 1] * factor + s % factor;
                outCoords[o * 4 + 2] = coords[i * 4 + 2] * factor + s / factor % factor;
                outCoords[o * 4 + 3] = coords[i * 4 + 3] * factor + s / (factor * factor) % factor;
                index[o++] = i * child + s;
            }
        return new Plan(outCoords, index, resolution * factor, child);
    }

    /// <summary>Merges each coarse voxel's children into channels: feats <c>[N, channels]</c> → <c>[1, M, ChildCount·channels]</c>.</summary>
    public static Tensor SpatialToChannelFeats(IBackend backend, Tensor feats, Plan plan, int channels)
    {
        RequireF32(feats, nameof(feats));
        int rows = RowCount(feats, channels, nameof(feats));
        if (plan.Index.Length != rows)
            throw new ArgumentException($"Plan maps {plan.Index.Length} input voxels, feats hold {rows}.", nameof(plan));
        int m = plan.Coords.Length / 4;
        // Scatter into the flat [m·ChildCount, channels] layout the row ops address, then view it as [1, m, ChildCount·channels].
        Tensor flat = new(new TensorShape(1, m * plan.ChildCount, channels), DType.F32);
        backend.Fill(flat, 0f);
        Tensor idx = IndexTensor.Rows(plan.Index);
        backend.ScatterAddRows(flat, feats, idx);
        backend.Sync();
        idx.Dispose();
        return flat.Reshape(new TensorShape(1, m, plan.ChildCount * channels));
    }

    /// <summary>Splits channel groups back onto children: feats <c>[N, ChildCount·channels]</c> → <c>[1, M, channels]</c>.</summary>
    public static Tensor ChannelToSpatialFeats(IBackend backend, Tensor childFeats, Plan plan, int channels)
    {
        RequireF32(childFeats, nameof(childFeats));
        if (channels <= 0 || childFeats.ElementCount % ((long)plan.ChildCount * channels) != 0)
        {
            throw new ArgumentException(
                $"Child feats of {childFeats.ElementCount} values do not split into {plan.ChildCount} children of {channels} channels.",
                nameof(childFeats));
        }
        int m = plan.Index.Length;
        long childRows = childFeats.ElementCount / channels;
        Tensor flat = childFeats.Reshape(new TensorShape(childRows, channels));
        Tensor o = new(new TensorShape(1, m, channels), DType.F32);
        Tensor idx = IndexTensor.Rows(plan.Index);
        backend.GatherRows(o, flat, idx);
        backend.Sync();
        idx.Dispose();
        return o;
    }

    /// <summary>Averages contiguous channel groups: <c>[..., K]</c> → <c>[1, rows, groups]</c>; group <c>o</c> spans <c>[o·g, (o+1)·g)</c>.</summary>
    /// <remarks>CPU-only: reads and writes host memory through <c>DataPointer</c>.</remarks>
    public static Tensor MeanChannelGroups(Tensor feats, int groups)
    {
        RequireF32(feats, nameof(feats));
        int k = (int)feats.Shape[feats.Shape.Rank - 1];
        if (groups <= 0 || k % groups != 0)
            throw new ArgumentException($"{k} channels do not split into {groups} equal groups.", nameof(groups));
        int rows = (int)(feats.ElementCount / k), g = k / groups;
        Tensor o = new(new TensorShape(1, rows, groups), DType.F32);
        float* s = (float*)feats.DataPointer, d = (float*)o.DataPointer;
        float inv = 1f / g;
        for (int r = 0; r < rows; r++)
            for (int oc = 0; oc < groups; oc++)
            {
                float acc = 0f;
                for (int j = 0; j < g; j++) acc += s[(long)r * k + oc * g + j];
                d[(long)r * groups + oc] = acc * inv;
            }
        return o;
    }

    /// <summary>Checks that coords are <c>(batch, x, y, z)</c> quads with batch ≥ 0 and spatial values in <c>[0, resolution)</c>.</summary>
    public static void ValidateCoords(int[] coords, int resolution)
    {
        if (resolution <= 0) throw new ArgumentOutOfRangeException(nameof(resolution), resolution, "Resolution must be positive.");
        if (coords.Length % 4 != 0)
            throw new ArgumentException("Coords must hold four ints (batch, x, y, z) per voxel.", nameof(coords));
        for (int i = 0; i < coords.Length; i += 4)
        {
            if (coords[i] < 0 || coords[i + 1] < 0 || coords[i + 2] < 0 || coords[i + 3] < 0
                || coords[i + 1] >= resolution || coords[i + 2] >= resolution || coords[i + 3] >= resolution)
            {
                throw new ArgumentException(
                    $"Voxel {i / 4} at ({coords[i]}, {coords[i + 1]}, {coords[i + 2]}, {coords[i + 3]}) is outside resolution {resolution}.",
                    nameof(coords));
            }
        }
    }

    private static int ChildCount(int factor) => factor * factor * factor;

    private static long CellKey(int b, int x, int y, int z, int extent)
        => (((long)b * extent + x) * extent + y) * extent + z;

    private static void DecodeCell(long key, int extent, int[] dst, int at)
    {
        dst[at + 3] = (int)(key % extent); key /= extent;
        dst[at + 2] = (int)(key % extent); key /= extent;
        dst[at + 1] = (int)(key % extent);
        dst[at] = (int)(key / extent);
    }

    private static int RowCount(Tensor feats, int channels, string name)
    {
        if (channels <= 0 || feats.ElementCount % channels != 0)
            throw new ArgumentException($"Feats of {feats.ElementCount} values do not split into rows of {channels} channels.", name);
        return (int)(feats.ElementCount / channels);
    }

    private static void RequireF32(Tensor t, string name)
    {
        if (t.DType != DType.F32)
            throw new NotSupportedException($"Sparse space-to-channel ops require F32 features; '{name}' is {t.DType}.");
    }

    private static void ValidateGeometry(int[] coords, int resolution, int factor)
    {
        if (factor < 2) throw new ArgumentOutOfRangeException(nameof(factor), factor, "Resample factor must be at least 2.");
        ValidateCoords(coords, resolution);
    }

    /// <summary>Output coords, row indices and the grid they live on, for one resample direction.</summary>
    /// <param name="Coords">Output voxel coords <c>[M·4]</c> as (batch, x, y, z); sorted by (batch, x, y, z) for space-to-channel.</param>
    /// <param name="Index">Per voxel row, <c>row·ChildCount + child</c>: input voxels for space-to-channel, output voxels for channel-to-spatial.</param>
    /// <param name="Resolution">Grid resolution of the output coords.</param>
    /// <param name="ChildCount">Children per coarse or parent voxel, <c>factor³</c>.</param>
    public readonly record struct Plan(int[] Coords, int[] Index, int Resolution, int ChildCount);
}
