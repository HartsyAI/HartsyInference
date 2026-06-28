using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.TextGen.Cli;

/// <summary>Mamba (SSM) smoke / validation harness. HARTSY_MAMBA_IDS feeds exact token ids; dumps the next-token
/// logits (argmax + top-5) and optionally writes them to HARTSY_MAMBA_DUMP for reference comparison.</summary>
public static class MambaRunner
{
    public static int Run(string backendName)
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        if (path is null || !File.Exists(path)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to a mamba .gguf."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));

        using MambaModel model = MambaModel.Load(path);
        Console.WriteLine($"=== mamba ({backendName}) {Path.GetFileName(path)} ===");
        Console.WriteLine($"  d_model={model.DModel} layers={model.NumLayers} d_inner={model.DInner} d_state={model.DState} conv_k={model.ConvKernel} dt_rank={model.DtRank} vocab={model.VocabSize}");

        string idsEnv = Environment.GetEnvironmentVariable("HARTSY_MAMBA_IDS") ?? "510,318,257";   // smoke ids
        int[] ids = idsEnv.Split(',').Select(s => int.Parse(s.Trim())).ToArray();
        Console.WriteLine($"  ids[{ids.Length}] = [{string.Join(",", ids)}]");

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
        return 0;
    }
}
