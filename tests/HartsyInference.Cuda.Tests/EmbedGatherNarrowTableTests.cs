using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Device-indexed row gather from an F16 or BF16 embed table (lm_embed_gather.cu) against the host widening of the
/// same rows. The table values have short mantissas, so both narrowing conversions are exact and the comparison is exact.</summary>
[Collection("CudaSerial")]
public sealed unsafe class EmbedGatherNarrowTableTests
{
    private const int Vocab = 16;
    private const int Hidden = 96;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmbedGatherDecodeStep_NarrowTable_MatchesHostWidening(bool bf16)
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);
        IBackend b = backend;
        float[] reference = new float[Vocab * Hidden];
        Tensor table = new(new TensorShape(Vocab, Hidden), bf16 ? DType.BF16 : DType.F16);
        Span<ushort> raw = bf16 ? table.AsSpan<ushort>() : default;
        Span<Half> half = bf16 ? default : table.AsSpan<Half>();
        for (int i = 0; i < reference.Length; i++)
        {
            float value = ((i * 37) % 201 - 100) * 0.25f;
            if (bf16)
            {
                ushort hi = (ushort)((uint)BitConverter.SingleToInt32Bits(value) >> 16);
                raw[i] = hi;
                reference[i] = BitConverter.UInt32BitsToSingle((uint)hi << 16);
            }
            else
            {
                half[i] = (Half)value;
                reference[i] = (float)(Half)value;
            }
        }

        b.PreloadWeights([table]);
        ulong token = b.AllocDeviceTokenId();
        try
        {
            foreach (int row in new[] { 0, 7, Vocab - 1 })
            {
                b.WriteDeviceTokenId(token, row);
                Tensor output = new(new TensorShape(1, 1, Hidden), DType.F32);
                b.EmbedGatherDecodeStep(output, table, token);
                ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();
                for (int c = 0; c < Hidden; c++)
                {
                    Assert.Equal(reference[row * Hidden + c], got[c]);
                }
                output.Dispose();
            }
        }
        finally
        {
            b.FreeDeviceTokenId(token);
            b.FreeWeights([table]);
            table.Dispose();
        }
    }
}
