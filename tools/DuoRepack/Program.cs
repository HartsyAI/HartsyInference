using System.Globalization;
using HartsyInference.ModelAssets.Duo;

const string HelpText = """
DuoRepack - make the Duo v1 Realization weights loadable without torch, and record a sha256 manifest of everything written.

  Sources are only read. Every command writes <out>/duo-repack-manifest.json (size and sha256 of each output, the licence note).
  The output of a Duo repack is a modified MiniMax H3 Model Derivative: keep the licence and NOTICE beside it and read
  docs/Research/DUO_V1_WEIGHTS.md before anything is shared. This tool uploads nothing.

USAGE
  DuoRepack lora --dcp <folder> --out <dir> [--family <prefix>]
      The released distributed-checkpoint LoRA as <dir>/duo_lora.safetensors (F32, diffusers module names, no pickle).
  DuoRepack merge --backbone <dir> --lora <dcp folder | duo_lora.safetensors> --out <dir> [--family <prefix>] [--alpha <n>]
                  [--input-qkv interleaved|blocks] [--drop-adaln] [--heads 56] [--head-dim 128]
      Merges the LoRA into every shard of the original-layout backbone (13 shards, about 66 GB BF16). The fused qkv comes out
      as [Q; K; V] blocks, the order the engine reads. --drop-adaln leaves out the 50 per-block adaln_proj layers (26 GB) for use
      with the tables below. Needs disk for a full copy of the backbone.
  DuoRepack tables --backbone <dir> --out <dir> (--stage distill|causal | --times 0,0.0625,...) [--precision bf16|f32]
      The exact AdaLN tables of a fixed time list as <dir>/duo_adaln_tables.safetensors. distill is 5 times, causal 51.
""";

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine(HelpText);
    return 0;
}

Dictionary<string, string> options = new(StringComparer.Ordinal);
for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--drop-adaln")
    {
        options[args[i]] = "true";
    }
    else if (args[i].StartsWith("--", StringComparison.Ordinal) && i + 1 < args.Length)
    {
        options[args[i]] = args[++i];
    }
    else
    {
        Console.Error.WriteLine($"Unexpected argument '{args[i]}'. Run with --help.");
        return 2;
    }
}

string Require(string name) => options.TryGetValue(name, out string? value) ? value : throw new ArgumentException($"{name} is required.");
Dictionary<string, string> notes = new()
{
    ["licence"] = "MiniMax H3 Community License; keep LICENSE and NOTICE beside every distributed file. See docs/Research/DUO_V1_WEIGHTS.md.",
    ["tool"] = "DuoRepack " + args[0],
};

try
{
    string outDir = Path.GetFullPath(Require("--out"));
    Directory.CreateDirectory(outDir);
    switch (args[0])
    {
        case "lora":
        {
            string dcp = Path.GetFullPath(Require("--dcp"));
            using DuoLoraSource lora = DuoLoraSource.FromDcp(dcp, options.GetValueOrDefault("--family"));
            DuoRepacker.WriteLoraSafetensors(lora, Path.Combine(outDir, "duo_lora.safetensors"));
            Console.WriteLine($"{lora.Modules.Count} LoRA modules written.");
            DuoRepackManifest.Write(outDir, ["duo_lora.safetensors"], new Dictionary<string, string> { ["dcp"] = dcp }, notes);
            break;
        }
        case "merge":
        {
            string backbone = Path.GetFullPath(Require("--backbone")), loraPath = Path.GetFullPath(Require("--lora"));
            using DuoLoraSource lora = Directory.Exists(loraPath)
                ? DuoLoraSource.FromDcp(loraPath, options.GetValueOrDefault("--family"))
                : DuoLoraSource.FromSafetensors(loraPath);
            DuoMergeOptions merge = new()
            {
                Alpha = options.TryGetValue("--alpha", out string? alpha) ? double.Parse(alpha, CultureInfo.InvariantCulture) : null,
                Heads = int.Parse(options.GetValueOrDefault("--heads", "56"), CultureInfo.InvariantCulture),
                HeadDim = int.Parse(options.GetValueOrDefault("--head-dim", "128"), CultureInfo.InvariantCulture),
                InputQkv = options.GetValueOrDefault("--input-qkv", "interleaved") switch
                {
                    "interleaved" => DuoQkvInput.HeadInterleaved,
                    "blocks" => DuoQkvInput.Blocks,
                    string other => throw new ArgumentException($"--input-qkv must be interleaved or blocks, not '{other}'."),
                },
                DropBlockAdalnProj = options.ContainsKey("--drop-adaln"),
            };
            IReadOnlyList<string> written = DuoRepacker.MergeBackbone(backbone, lora, outDir, merge, Console.WriteLine);
            DuoRepackManifest.Write(outDir, written,
                new Dictionary<string, string> { ["backbone"] = backbone, ["lora"] = loraPath }, notes);
            break;
        }
        case "tables":
        {
            string backbone = Path.GetFullPath(Require("--backbone"));
            float[] times = options.GetValueOrDefault("--stage") switch
            {
                "distill" => DuoSchedule.Grid(DuoSchedule.DistillSteps, DuoSchedule.DistillShift),
                "causal" => DuoSchedule.Grid(DuoSchedule.CausalSteps, DuoSchedule.CausalShift),
                null => [.. Require("--times").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(t => float.Parse(t, CultureInfo.InvariantCulture))],
                string other => throw new ArgumentException($"--stage must be distill or causal, not '{other}'."),
            };
            DuoAdalnPrecision precision = options.GetValueOrDefault("--precision", "bf16") switch
            {
                "bf16" => DuoAdalnPrecision.Bf16Reference,
                "f32" => DuoAdalnPrecision.F32,
                string other => throw new ArgumentException($"--precision must be bf16 or f32, not '{other}'."),
            };
            DuoRepacker.WriteAdalnTables(backbone, times, Path.Combine(outDir, "duo_adaln_tables.safetensors"), precision, Console.WriteLine);
            DuoRepackManifest.Write(outDir, ["duo_adaln_tables.safetensors"], new Dictionary<string, string> { ["backbone"] = backbone }, notes);
            break;
        }
        default:
            Console.Error.WriteLine($"Unknown command '{args[0]}'. Run with --help.");
            return 2;
    }
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidDataException or IOException or NotSupportedException or FormatException or KeyNotFoundException)
{
    Console.Error.WriteLine(error.Message);
    return 1;
}
