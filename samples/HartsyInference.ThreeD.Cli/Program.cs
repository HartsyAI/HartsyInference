using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Cli;

/// <summary>Demo CLI: load a Hunyuan3D-2 shape checkpoint from a local directory, generate a mesh from a
/// single PNG image, and write a <c>.glb</c>. Living documentation for the image→3D path; production code
/// should call <see cref="Hunyuan3DShapePipeline"/> directly.
/// <para><b>Numerics validation-pending</b> — the pipeline is structurally complete but per-checkpoint
/// fidelity awaits the reference-diff pass, so output geometry is not yet trustworthy.</para></summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        string? imagePath = null, modelPath = null, outPath = "output.glb", backendName = "cpu", type = "hunyuan3d";
        int steps = 0, grid = 0; int? seed = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-i" or "--image": imagePath = args[++i]; break;
                case "-m" or "--model": modelPath = args[++i]; break;
                case "-o" or "--out": outPath = args[++i]; break;
                case "-b" or "--backend": backendName = args[++i]; break;
                case "-t" or "--type": type = args[++i].ToLowerInvariant(); break;
                case "--steps": steps = int.Parse(args[++i]); break;
                case "--grid": grid = int.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                default: Console.Error.WriteLine($"unknown flag: {args[i]}"); return 1;
            }
        }

        if (imagePath is null || modelPath is null)
        {
            Console.Error.WriteLine("missing required --image and/or --model");
            PrintUsage();
            return 1;
        }
        if (!File.Exists(imagePath)) { Console.Error.WriteLine($"image not found: {imagePath}"); return 1; }

        Console.Error.WriteLine($"image:   {imagePath}");
        Console.Error.WriteLine($"model:   {modelPath}");
        Console.Error.WriteLine($"backend: {backendName}");

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(imagePath);

        using IBackend backend = CreateBackend(backendName);

        ImageTo3DRequest req = new()
        {
            ImageRgb = rgb, Width = w, Height = h,
            Steps = steps, GridResolution = grid, Seed = seed,
        };
        void Progress(GenerationProgress p) => Console.Error.Write($"\r  step {p.Step}/{p.TotalSteps}");

        Stopwatch sw = Stopwatch.StartNew();
        Console.Error.WriteLine($"loading {type} checkpoint...");
        ThreeDResult result;
        switch (type)
        {
            case "hunyuan3d":
            {
                using Hunyuan3DShapePipeline pipeline = Hunyuan3DShapePipeline.LoadFromPath(backend, modelPath);
                Console.Error.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");
                sw.Restart(); Console.Error.WriteLine("generating...");
                result = pipeline.Generate(req, Progress);
                break;
            }
            case "triposr":
            {
                using TripoSrPipeline pipeline = TripoSrPipeline.LoadFromPath(backend, modelPath);
                Console.Error.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");
                sw.Restart(); Console.Error.WriteLine("generating...");
                result = pipeline.Generate(req, Progress);
                break;
            }
            default:
                Console.Error.WriteLine($"unknown --type '{type}' (use hunyuan3d or triposr)");
                return 1;
        }
        Console.Error.WriteLine($"\n  generated in {sw.Elapsed.TotalSeconds:F1}s (seed {result.Seed})");

        if (result.Mesh is null || result.Mesh.TriangleCount == 0)
        {
            Console.Error.WriteLine("no surface extracted (empty mesh) — check iso level / checkpoint.");
            return 2;
        }

        GlbWriter.Save(outPath, result.Mesh);
        Console.Error.WriteLine($"wrote {outPath}  ({result.Mesh.VertexCount} verts, {result.Mesh.TriangleCount} tris)");
        return 0;
    }

    private static IBackend CreateBackend(string name) => name.ToLowerInvariant() switch
    {
        "cuda" => new HartsyInference.Cuda.CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx")),
        "cpu" => new HartsyInference.Cpu.CpuBackend(),
        _ => throw new ArgumentException($"unknown backend '{name}' (use cpu or cuda)"),
    };

    private static void PrintUsage()
    {
        Console.Error.WriteLine(
            """
            hartsyinference-3d — Hunyuan3D-2 image → mesh (.glb)

            Usage:
              hartsyinference-3d -i <image.png> -m <model-dir> [-t hunyuan3d|triposr] [-o out.glb] [options]

            Options:
              -i, --image <png>     input conditioning image (PNG)
              -m, --model <dir>     directory containing the model .safetensors
              -t, --type <name>     hunyuan3d | triposr  (default: hunyuan3d)
              -o, --out <glb>       output path (default: output.glb)
              -b, --backend <name>  cpu | cuda  (default: cpu)
                  --steps <n>       denoise steps — Hunyuan3D only (default: model config)
                  --grid <n>        marching-cubes grid resolution (default: model config)
                  --seed <n>        RNG seed (default: random)
            """);
    }
}
