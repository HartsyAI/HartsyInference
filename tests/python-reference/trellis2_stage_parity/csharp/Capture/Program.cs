using System.Diagnostics;
using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;

// usage: t2cap <modelsFolder> <rgba> <grid> <ptxDir> <outDir> <steps> [shape]
string folder = args[0], imagePath = args[1], ptx = args[3], outDir = args[4];
int grid = int.Parse(args[2]), steps = int.Parse(args[5]);
bool shapeOnly = args.Length > 6 && args[6] == "shape";
byte[] rgba = File.ReadAllBytes(imagePath);
int size = (int)Math.Sqrt(rgba.Length / 4);
if (rgba.Length != size * size * 4) throw new ArgumentException($"'{imagePath}' is not a square raw RGBA image ({rgba.Length} bytes).");
byte[] rgb = new byte[size * size * 3], alpha = new byte[size * size];
for (int i = 0; i < size * size; i++) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]; alpha[i] = rgba[i * 4 + 3]; }
Directory.CreateDirectory(outDir);
File.Delete(Path.Combine(outDir, "times.txt"));
Stopwatch t0 = Stopwatch.StartNew();
using CudaBackend backend = new(0, ptx);
using Trellis2ImageTo3DPipeline pipeline = Trellis2ImageTo3DPipeline.LoadFromPath(backend, folder);
Stage.Out = outDir; Stage.Backend = backend;

void Swap(string field, Func<object, object> wrap)
{
    FieldInfo f = typeof(Trellis2ImageTo3DPipeline).GetField(field, BindingFlags.NonPublic | BindingFlags.Instance)!;
    f.SetValue(pipeline, wrap(f.GetValue(pipeline)!));
}
Swap("_shape", o => new ShapeSpy((ITrellis2ShapeFlowStage)o));
object decoder = typeof(Trellis2ImageTo3DPipeline).GetField("_decoder", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(pipeline)!;
DecoderSpy dspy = new((ISparseShapeSubdivisionDecoder)decoder);
Swap("_decoder", _ => dspy);
Swap("_mesher", o => new MesherSpy((IDualGridMesher)o));
if (!shapeOnly)
{
    Swap("_textureFlow", o => new TexFlowSpy((ITrellis2TextureFlowStage)o));
    Swap("_textureDecoder", o => new TexDecSpy((ITrellis2TextureDecoderStage)o));
}

Trellis2InputImage image = new() { Rgb = rgb, Alpha = alpha, Width = size, Height = size };
double lastMs = 0;
void Progress(GenerationProgress p) { Console.WriteLine($"[{t0.Elapsed.TotalSeconds:F0}s] step {p.Step}/{p.TotalSteps} (+{p.ElapsedMs - lastMs:F0} ms)"); lastMs = p.ElapsedMs; }
Trellis2Options opt = new() { Grid = grid, Seed = 1, Texture = !shapeOnly, Steps = steps };
if (shapeOnly) { Mesh m = pipeline.Generate(image, opt, Progress); Console.WriteLine($"done shape-only: {m.VertexCount} verts"); }
else { PbrPostProcessResult r = pipeline.GenerateTextured(image, opt, Progress, CancellationToken.None); Console.WriteLine($"done textured: {r.Mesh.VertexCount} verts"); }
Console.WriteLine($"TOTAL {t0.Elapsed.TotalSeconds:F0} s");

static class Stage
{
    public static string Out = "";
    public static IBackend Backend = null!;
    public static float[] Host(Tensor t) { Backend.OffloadActivation(t); Backend.Sync(); return t.AsReadOnlySpan<float>().ToArray(); }
    public static (string, Array, long[])[] Sparse(string name, SparseTensor s) =>
    [
        (name + "_coords", s.Coords, [s.Count, 4]),
        (name + "_feats", Host(s.Feats), [s.Count, s.Channels]),
    ];
    public static (string, Array, long[]) Dense(string name, Tensor t) => (name, Host(t), Enumerable.Range(0, t.Shape.Rank).Select(i => (long)t.Shape[i]).ToArray());
    public static void Time(string label, double ms) { Console.WriteLine($"STAGE {label}: {ms / 1000:F2} s"); File.AppendAllText(Path.Combine(Out, "times.txt"), $"{label}\t{ms / 1000:F3}\n"); }
}

sealed class ShapeSpy(ITrellis2ShapeFlowStage inner) : ITrellis2ShapeFlowStage
{
    public SparseTensor Sample(IBackend b, Trellis2ShapeFlowKind kind, int[] coords, int grid, Tensor cond, Tensor neg, int seed, Trellis2SamplerParams p, Action<int>? onStep)
    {
        Stopwatch sw = Stopwatch.StartNew();
        SparseTensor r = inner.Sample(b, kind, coords, grid, cond, neg, seed, p, onStep);
        b.Sync();
        Stage.Time($"shape_flow_{kind}", sw.Elapsed.TotalMilliseconds);
        Tns.Write(Path.Combine(Stage.Out, $"shape_{kind}.tns"), [.. Stage.Sparse("latent", r), Stage.Dense("cond", cond), ("voxels", coords, [coords.Length / 4, 4])]);
        return r;
    }
}

sealed class DecoderSpy(ISparseShapeSubdivisionDecoder inner) : ISparseShapeSubdivisionDecoder, ISparseShapeUpsampler
{
    public DualGridField Decode(IBackend b, SparseTensor latent) => Decode(b, latent, new ShapeVaeSubdivisions());
    public DualGridField Decode(IBackend b, SparseTensor latent, ShapeVaeSubdivisions rec)
    {
        Stopwatch sw = Stopwatch.StartNew();
        DualGridField f = inner.Decode(b, latent, rec);
        b.Sync();
        Stage.Time("shape_decode", sw.Elapsed.TotalMilliseconds);
        List<(string, Array, long[])> items = [.. Stage.Sparse("latent", latent)];
        DualGridVoxels v = f.Voxels; int n = v.Coords.Length / 3;
        items.Add(("coords", v.Coords, [n, 3])); items.Add(("dual", v.DualVertices, [n, 3]));
        items.Add(("inter", v.Intersected.Select(x => x ? 1 : 0).ToArray(), [n, 3])); items.Add(("split", f.SplitWeight, [n]));
        for (int i = 0; i < rec.Count; i++) items.Add(Stage.Dense($"subdiv{i}", rec.Level(i)));
        Tns.Write(Path.Combine(Stage.Out, "shape_decode.tns"), [.. items]);
        return f;
    }
    public int[] Upsample(IBackend b, SparseTensor latent, int times) => inner is ISparseShapeUpsampler u ? u.Upsample(b, latent, times) : throw new NotSupportedException();
}

sealed class MesherSpy(IDualGridMesher inner) : IDualGridMesher
{
    public Mesh Extract(DualGridField field, SceneGrid grid)
    {
        Stopwatch sw = Stopwatch.StartNew();
        Mesh m = inner.Extract(field, grid);
        Stage.Time("mesher", sw.Elapsed.TotalMilliseconds);
        Tns.Write(Path.Combine(Stage.Out, "mesh.tns"), ("vertices", m.Vertices, [m.VertexCount, 3]), ("indices", m.Indices, [m.TriangleCount, 3]),
            ("grid", new[] { grid.Resolution }, [1]), ("origin", new[] { grid.Origin.X, grid.Origin.Y, grid.Origin.Z }, [3]), ("voxel", new[] { grid.VoxelSize }, [1]));
        return m;
    }
}

sealed class TexFlowSpy(ITrellis2TextureFlowStage inner) : ITrellis2TextureFlowStage
{
    public bool Supports(Trellis2ShapeFlowKind kind) => inner.Supports(kind);
    public SparseTensor Sample(IBackend b, Trellis2ShapeFlowKind kind, SparseTensor shape, Tensor cond, Tensor neg, int seed, Trellis2SamplerParams p, Action<int>? onStep)
    {
        Stopwatch sw = Stopwatch.StartNew();
        SparseTensor r = inner.Sample(b, kind, shape, cond, neg, seed, p, onStep);
        b.Sync();
        Stage.Time("texture_flow", sw.Elapsed.TotalMilliseconds);
        Tns.Write(Path.Combine(Stage.Out, "texture_flow.tns"), [.. Stage.Sparse("shape", shape), Stage.Dense("cond", cond), .. Stage.Sparse("tex_latent", r)]);
        return r;
    }
}

sealed class TexDecSpy(ITrellis2TextureDecoderStage inner) : ITrellis2TextureDecoderStage
{
    public SparseTensor Decode(IBackend b, SparseTensor latent, ShapeVaeSubdivisions guides)
    {
        Stopwatch sw = Stopwatch.StartNew();
        SparseTensor r = inner.Decode(b, latent, guides);
        b.Sync();
        Stage.Time("texture_decode", sw.Elapsed.TotalMilliseconds);
        Tns.Write(Path.Combine(Stage.Out, "texture_decode.tns"), [.. Stage.Sparse("latent", latent), .. Stage.Sparse("out", r)]);
        return r;
    }
}
