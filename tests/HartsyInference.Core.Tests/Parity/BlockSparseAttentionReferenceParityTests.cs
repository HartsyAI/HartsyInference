using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tests.Backends;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Parity;

/// <summary>Every output element of the CPU's block-sparse attention against the full outputs LongCat-Video's kernel wrote
/// (tests/python-reference/longcat_bsa/dump_longcat_bsa.py). Skipped, and so not evidence, until HARTSY_LONGCAT_BSA_DIR names the
/// dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class BlockSparseAttentionReferenceParityTests(ITestOutputHelper log)
{
    private const string EnvName = "HARTSY_LONGCAT_BSA_DIR";

    [ReferenceDumpFact]
    public void EveryOutputElementMatchesTheReferenceKernel()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using CpuBackend backend = new();
        foreach (BlockSparseAttentionCase reference in BlockSparseAttentionCase.All)
        {
            float[] expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(dir, reference.Name + ".output.f32"))).ToArray();
            int[] blocks = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(dir, reference.Name + ".selected.i32"))).ToArray();

            (float[] output, int[] selected) = reference.Run(backend);

            Assert.Equal(blocks, selected);
            double error = BlockSparseAttentionCase.RelativeRms(output, expected);
            log.WriteLine($"{reference.Name}: relative RMS {error:E3} over {expected.Length} values, {blocks.Length} selected blocks equal");
            Assert.True(error <= 1e-5, $"{reference.Name}: relative RMS {error:E3}");
        }
    }

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ReferenceDumpFactAttribute : FactAttribute
    {
        public ReferenceDumpFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "report.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no report.json; the block-sparse attention reference comparison did not run.";
        }
    }
}
