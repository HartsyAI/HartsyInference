using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

// TEMPORARY diagnostic (delete before merge): splits the CUDA prefill error into TF32 GEMM, TF32 SDPA and a path bug by per-layer streams.
public sealed class Glm5NextCudaDiagnosticTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny");

    [CudaFact]
    public void Diagnose_PrefillErrorByTf32Lane()
    {
        try
        {
            Lane("cpu", null, null);
            Lane("cuda default (TF32 GEMM, TF32 SDPA)", false, false);
            Lane("cuda exact GEMM, TF32 SDPA", true, false);
            Lane("cuda TF32 GEMM, exact SDPA", false, true);
            Lane("cuda exact GEMM, exact SDPA", true, true);
        }
        finally
        {
            KnobStore.ResetOverrides();
        }
    }

    private void Lane(string name, bool? exactGemm, bool? exactSdpa)
    {
        KnobStore.ResetOverrides();
        if (exactGemm is { } g) KnobStore.Set(EngineKnobs.HighPrecisionGemm, g);
        if (exactSdpa is { } s) KnobStore.Set(EngineKnobs.SdpaNoTf32, s);
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        float[] logits = Run(name == "cpu" ? new CpuBackend() : new CudaBackend(0, CudaPtxDirectory.Resolve()), c, ids, out float[] streamsDiffs);
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "logits.npy")).Data;
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - logits[i]));
        output.WriteLine($"{name}: logits max abs {worst:E3}");
        output.WriteLine($"  per-layer stream max abs (layer 0..{streamsDiffs.Length - 1}): {string.Join(" ", streamsDiffs.Select(v => v.ToString("E2")))}");
    }

    private static float[] Run(IBackend backend, Glm5NextConfig c, int[] ids, out float[] streamDiffs)
    {
        using (backend)
        using (BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType))
        using (Glm5NextModel model = new(backend, c, source))
        using (Glm5NextState state = new(c, ids.Length))
        {
            List<float> diffs = [];
            model.LayerObserver = (layer, stream) =>
            {
                float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, $"layer_{layer}_stream.npy")).Data;
                float worst = 0f;
                for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - stream[i]));
                diffs.Add(worst);
            };
            float[] logits = model.Forward(ids, state);
            streamDiffs = [.. diffs];
            return logits;
        }
    }
}
