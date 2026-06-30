using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.TextGen.Cli;

/// <summary>Mamba-2 (SSD) smoke / validation harness. HARTSY_MAMBA_IDS feeds exact token ids; dumps the
/// next-token logits (argmax + top-5) and optionally writes them to HARTSY_MAMBA_DUMP for reference comparison.
/// With HARTSY_MAMBA_GEN=N it greedy-generates N tokens (re-running the stack each step) and prints the ids.</summary>
public static class Mamba2Runner
{
    public static int Run(string backendName)
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        if (path is null || !File.Exists(path)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to a mamba2 .gguf."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));

        using Mamba2Model model = Mamba2Model.Load(path);
        Console.WriteLine($"=== mamba2 ({backendName}) {Path.GetFileName(path)} ===");
        Console.WriteLine($"  d_model={model.DModel} layers={model.NumLayers} d_inner={model.DInner} d_state={model.DState} conv_k={model.ConvKernel} heads={model.NumHeads}x{model.HeadDim} groups={model.NumGroups} vocab={model.VocabSize}");

        string idsEnv = Environment.GetEnvironmentVariable("HARTSY_MAMBA_IDS") ?? "510,318,257";
        List<int> ids = [.. idsEnv.Split(',').Select(s => int.Parse(s.Trim()))];
        Console.WriteLine($"  ids[{ids.Count}] = [{string.Join(",", ids)}]");

        float[] logits = model.ForwardLastLogits(backend, ids);
        int argmax = 0; for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        (int i, float v)[] top5 = logits.Select((v, i) => (i, v)).OrderByDescending(t => t.v).Take(5).ToArray();
        Console.WriteLine($"  argmax token = {argmax} (logit {logits[argmax]:F4})");
        Console.WriteLine($"  top5 = [{string.Join(", ", top5.Select(t => $"{t.i}:{t.v:F3}"))}]");

        string? dump = Environment.GetEnvironmentVariable("HARTSY_MAMBA_DUMP");
        if (dump is not null)
        {
            byte[] bytes = new byte[logits.Length * 4];
            Buffer.BlockCopy(logits, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(dump, bytes);
            Console.WriteLine($"  dumped logits → {dump}");
        }

        int gen = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_MAMBA_GEN"), out int g) ? g : 0;
        if (gen > 0)
        {
            List<int> outIds = [];
            for (int step = 0; step < gen; step++)
            {
                float[] lg = model.ForwardLastLogits(backend, ids);
                int am = 0; for (int i = 1; i < lg.Length; i++) if (lg[i] > lg[am]) am = i;
                outIds.Add(am); ids.Add(am);
            }
            Console.WriteLine($"  generated ids = [{string.Join(",", outIds)}]");
        }
        return 0;
    }
}
