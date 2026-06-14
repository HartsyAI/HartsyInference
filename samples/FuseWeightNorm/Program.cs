using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Samples.FuseWeightNorm;

/// <summary>Offline weight-norm fusion tool. Scans an input safetensors file for any
/// <c>*.weight_g</c> / <c>*.weight_v</c> pair, fuses them into a single <c>*.weight</c>
/// tensor via <see cref="WeightNormFusion"/>, and writes a new safetensors file with
/// the fused weights and all other tensors copied through unchanged.
///
/// <para>Runtime fusion via <see cref="WeightNormFusion.Fuse"/> still works on the
/// audio codecs (EnCodec / DAC / SNAC / WavTokenizer / BiCodec / Mimi). This tool is for
/// production deployments that want to skip the load-time fusion cost — pre-fused
/// safetensors load instantly with the same key conventions any vanilla <c>nn.Conv1d</c>
/// would use.</para>
///
/// <para>Usage:</para>
/// <code>fuse-weight-norm INPUT.safetensors OUTPUT.safetensors</code></summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("Usage: fuse-weight-norm INPUT.safetensors OUTPUT.safetensors");
            Console.Error.WriteLine("");
            Console.Error.WriteLine("Scans the input file for *.weight_g / *.weight_v pairs and writes a new");
            Console.Error.WriteLine("safetensors file with each pair fused into a single *.weight tensor.");
            return 1;
        }

        string inputPath = args[0];
        string outputPath = args[1];
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            return 1;
        }

        Console.WriteLine($"Loading: {inputPath}");
        using SafeTensorsLoader loader = new();
        loader.Load(inputPath);
        Dictionary<string, Tensor> allTensors = loader.GetAllTensors();
        Console.WriteLine($"Loaded {allTensors.Count} tensors.");

        // Pair up weight_g / weight_v keys.
        Dictionary<string, Tensor> output = new();
        List<string> consumed = new();
        int fusedCount = 0;

        foreach (KeyValuePair<string, Tensor> kv in allTensors)
        {
            if (!kv.Key.EndsWith(".weight_g")) continue;
            string baseKey = kv.Key[..^".weight_g".Length];
            string vKey = baseKey + ".weight_v";
            string fusedKey = baseKey + ".weight";

            if (!allTensors.TryGetValue(vKey, out Tensor? v))
            {
                Console.Error.WriteLine($"Warning: found {kv.Key} but no companion {vKey}; skipping.");
                continue;
            }

            // Cast both to F32 if needed (mirrors WhisperOps.EnsureF32 path used at runtime).
            Tensor g32 = kv.Value.DType == DType.F32 ? kv.Value : kv.Value.CastTo(DType.F32);
            Tensor v32 = v.DType == DType.F32 ? v : v.CastTo(DType.F32);

            // Conv2d/1d/Linear: out_channels along axis 0 → use the generic WeightNormFusion.
            // ConvTranspose1d/2d: out_channels along axis 1 → use WeightNormFusionT.
            // The decision is based on whether the leading axis matches weight_g's element
            // count or whether axis 1 does.
            int axis0 = (int)v32.Shape[0];
            int axis1 = v32.Shape.Rank >= 2 ? (int)v32.Shape[1] : -1;
            long gCount = g32.ElementCount;

            Tensor fused;
            if (gCount == axis0)
            {
                fused = WeightNormFusion.Fuse(g32, v32);
            }
            else if (gCount == axis1 && v32.Shape.Rank == 3)
            {
                // ConvTranspose1d case.
                fused = Models.Codecs.EnCodec.WeightNormFusionT.Fuse(g32, v32);
            }
            else
            {
                Console.Error.WriteLine($"Warning: cannot infer axis for {baseKey}: weight_g={gCount} elements, weight_v shape={v32.Shape}; skipping.");
                if (g32 != kv.Value) g32.Dispose();
                if (v32 != v) v32.Dispose();
                continue;
            }

            output[fusedKey] = fused;
            consumed.Add(kv.Key);
            consumed.Add(vKey);
            fusedCount++;

            if (g32 != kv.Value) g32.Dispose();
            if (v32 != v) v32.Dispose();
        }

        // Copy through all non-consumed tensors.
        HashSet<string> consumedSet = [.. consumed];
        foreach (KeyValuePair<string, Tensor> kv in allTensors)
        {
            if (!consumedSet.Contains(kv.Key)) output[kv.Key] = kv.Value;
        }

        Console.WriteLine($"Fused {fusedCount} weight-norm pairs.");
        Console.WriteLine($"Writing: {outputPath} ({output.Count} tensors total)");
        SafeTensorsWriter.Save(outputPath, output);
        Console.WriteLine("Done.");
        return 0;
    }
}
