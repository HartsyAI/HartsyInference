using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Core.Tests;

/// <summary>A large <see cref="Tensor.CastTo"/> or <see cref="Tensor.DequantFp8E4M3ScaledToF16"/> is split into ranges
/// through <see cref="CpuParallel"/>. Its bytes match the same values cast through the serial loop, in pieces under the
/// split threshold, whether it fanned out over every core, ran under a <c>numerics.cpuThreads</c> cap of 1, or ran
/// inside an <see cref="CpuParallel.InlineScope"/>. And a large cast inside a parallel loop, which nests one fan-out in
/// another on the same capped scheduler, finishes at every cap, from two threads at once, inline, and from inside a raw
/// <see cref="Parallel.ForEach{TSource}(IEnumerable{TSource}, ParallelOptions, Action{TSource})"/> as the CPU path of
/// GPT-OSS's expert loop makes it. Serialized with the other classes that change process-wide knobs.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class TensorCastScheduleTests
{
    /// <summary>Over the 2^20-element split threshold, and not a whole number of 2^18-element ranges, so the last range
    /// is short.</summary>
    private const int Count = (1 << 20) + 4099;

    /// <summary>Under the threshold, so every reference piece goes through the serial loop and never through a range.</summary>
    private const int ReferencePiece = 1 << 19;

    private const float Fp8Scale = 0.37f;

    [Theory]
    [InlineData("F32", "F16")]
    [InlineData("BF16", "F32")]
    [InlineData("F8E4M3", "F32")]
    [InlineData("F32", "F8E4M3")]
    [InlineData("F32", "F8E5M2")]
    [InlineData("F64", "F32")]
    public void CastTo_MatchesTheSerialCastByteForByte_UnderEverySchedule(string fromName, string toName)
    {
        DType from = Parse(fromName), to = Parse(toName);
        using Tensor source = RandomTensor(from, Count, seed: fromName.Length * 31 + toName.Length);
        byte[] expected = InPieces(source, to.SizeInBytes, piece => piece.CastTo(to));

        (byte[] parallel, byte[] capped, byte[] inline) = UnderEverySchedule(() => Bytes(source.CastTo(to)));

        AssertSameBytes(expected, parallel, $"{fromName}->{toName}, every core");
        AssertSameBytes(expected, capped, $"{fromName}->{toName}, cap 1");
        AssertSameBytes(expected, inline, $"{fromName}->{toName}, inline");
    }

    [Fact]
    public void DequantFp8E4M3ScaledToF16_MatchesTheSerialDequantByteForByte_UnderEverySchedule()
    {
        using Tensor source = RandomTensor(DType.F8E4M3, Count, seed: 7);
        byte[] expected = InPieces(source, sizeof(ushort), piece => piece.DequantFp8E4M3ScaledToF16(Fp8Scale));

        (byte[] parallel, byte[] capped, byte[] inline) =
            UnderEverySchedule(() => Bytes(source.DequantFp8E4M3ScaledToF16(Fp8Scale)));

        AssertSameBytes(expected, parallel, "every core");
        AssertSameBytes(expected, capped, "cap 1");
        AssertSameBytes(expected, inline, "inline");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void CastsInsideAParallelLoop_Complete_AtEveryCap(int cap)
    {
        int threads = cap == 0 ? Environment.ProcessorCount : cap;
        WithCpuThreads(threads, () => RunBounded(CastInsideAParallelLoop));
    }

    /// <summary>A thread outside <see cref="CpuParallel"/>'s scheduler cannot run the cast's ranges itself, so each
    /// <see cref="Parallel.ForEach{TSource}(IEnumerable{TSource}, ParallelOptions, Action{TSource})"/> worker waits on
    /// the scheduler's own threads; this pins that the shape still finishes, with every cast intact.</summary>
    [Fact]
    public void CastsInsideARawParallelForEach_Complete()
    {
        WithCpuThreads(0, () => RunBounded(() =>
        {
            using Tensor source = RandomTensor(DType.F32, Count, seed: 11);
            byte[] expected = Bytes(source.CastTo(DType.BF16));
            const int Items = 12;
            bool[] intact = new bool[Items];
            ParallelOptions options = new() { MaxDegreeOfParallelism = Math.Clamp(Environment.ProcessorCount / 2, 1, 8) };
            Parallel.ForEach(Enumerable.Range(0, Items), options, i =>
            {
                intact[i] = Bytes(source.CastTo(DType.BF16)).AsSpan().SequenceEqual(expected);
            });
            Assert.DoesNotContain(false, intact);
        }));
    }

    /// <summary>An outer loop that fans out, each of whose iterations makes a cast big enough to fan out its own
    /// ranges; repeated so a race that only sometimes wedges gets several chances.</summary>
    private static void CastInsideAParallelLoop()
    {
        const int Iterations = 6;
        using Tensor source = RandomTensor(DType.F32, Count, seed: 5);
        byte[] expected = Bytes(source.CastTo(DType.BF16));
        for (int round = 0; round < 3; round++)
        {
            bool[] intact = new bool[Iterations];
            CpuParallel.For(Iterations, CpuParallel.MinWorkForParallel * Iterations, i =>
            {
                intact[i] = Bytes(source.CastTo(DType.BF16)).AsSpan().SequenceEqual(expected);
            });
            Assert.DoesNotContain(false, intact);
        }
    }

    /// <summary>Converts <paramref name="source"/> a piece at a time, each piece under the split threshold, and joins
    /// the results.</summary>
    private static byte[] InPieces(Tensor source, int outputElementSize, Func<Tensor, Tensor> convert)
    {
        int inputElementSize = source.DType.SizeInBytes;
        byte[] joined = new byte[(long)Count * outputElementSize];
        ReadOnlySpan<byte> all = source.AsReadOnlySpan<byte>();
        for (int start = 0; start < Count; start += ReferencePiece)
        {
            int length = Math.Min(ReferencePiece, Count - start);
            using Tensor piece = new(new TensorShape(length), source.DType);
            all.Slice(start * inputElementSize, length * inputElementSize).CopyTo(piece.AsSpan<byte>());
            piece.Fp8ScaleFactor = source.Fp8ScaleFactor;
            using Tensor converted = convert(piece);
            converted.AsReadOnlySpan<byte>().CopyTo(joined.AsSpan(start * outputElementSize));
        }
        return joined;
    }

    /// <summary>Random bytes, so every bit pattern turns up: NaNs, infinities, subnormals and both zeros included. An
    /// fp8 source carries a per-tensor scale, which the cast folds into every value.</summary>
    private static Tensor RandomTensor(DType dtype, int count, int seed)
    {
        Tensor tensor = new(new TensorShape(count), dtype);
        new Random(seed).NextBytes(tensor.AsSpan<byte>());
        if (dtype == DType.F8E4M3 || dtype == DType.F8E5M2) tensor.Fp8ScaleFactor = Fp8Scale;
        return tensor;
    }

    private static byte[] Bytes(Tensor tensor)
    {
        using (tensor)
        {
            return tensor.AsReadOnlySpan<byte>().ToArray();
        }
    }

    private static void AssertSameBytes(byte[] expected, byte[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int same = expected.AsSpan().CommonPrefixLength(actual);
        Assert.True(same == expected.Length, $"{what}: first difference at byte {same} of {expected.Length}");
    }

    private static DType Parse(string name) => name switch
    {
        "F64" => DType.F64,
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        "F8E4M3" => DType.F8E4M3,
        "F8E5M2" => DType.F8E5M2,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "not a cast dtype here"),
    };
}
