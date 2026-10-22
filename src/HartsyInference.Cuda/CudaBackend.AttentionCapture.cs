using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// diagnostics.attentionCaptureDir: real attention operands written to disk for offline kernel accuracy work.
public sealed partial class CudaBackend
{
    private const int CaptureMinTokens = 1024, CaptureHeads = 4, CaptureQueryRows = 512;
    private readonly Dictionary<string, int> _captureOrdinals = new(StringComparer.Ordinal);
    private readonly object _captureGate = new();
    private int _captureSkipNoted;

    /// <summary>Counts the call against its shape in <c>calls.tsv</c> and, when its ordinal is listed, writes batch 0's Q (four
    /// evenly spaced heads, up to 512 evenly spaced rows), K and V (those heads' key/value heads, every valid key) in the call's
    /// dtype with a JSON description. Skipped while the stream captures: a graph cannot read back. A failed write is logged and
    /// the call runs on.</summary>
    private void CaptureAttentionCall(in AttentionCall call, string dir)
    {
        if (Math.Min(call.QueryTokens, call.ValidKvLength) < CaptureMinTokens) return;
        if (StreamIsCapturing())
        {
            if (Interlocked.Exchange(ref _captureSkipNoted, 1) == 0)
                Logs.Warning("[attention capture] calls inside a captured graph are neither listed nor written; set numerics.ditGraph=false.");
            return;
        }
        string shape = string.Create(CultureInfo.InvariantCulture,
            $"{call.Query.DType}-b{call.Batch}-h{call.Heads}x{call.KvHeads}-q{call.QueryTokens}-k{call.ValidKvLength}-d{call.QkHeadDim}-{call.Query.Layout}");
        int ordinal;
        lock (_captureGate)
        {
            ordinal = _captureOrdinals.TryGetValue(shape, out int seen) ? seen : 0;
            _captureOrdinals[shape] = ordinal + 1;
            try
            {
                Directory.CreateDirectory(dir);
                File.AppendAllText(Path.Combine(dir, "calls.tsv"), string.Create(CultureInfo.InvariantCulture,
                    $"{shape}\t{ordinal}\t{call.Route}\t{(call.Mask.Bias is null ? "none" : "bias")}\t{call.Scale:R}\n"));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                Logs.Warning($"[attention capture] {shape} call {ordinal} not listed: {ex.Message}");
                return;
            }
        }
        if (!CaptureWanted(EngineKnobs.AttentionCaptureCalls.Value, shape, ordinal)) return;
        try
        {
            WriteCapture(in call, dir, shape, ordinal);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            Logs.Warning($"[attention capture] {shape} call {ordinal} not written: {ex.Message}");
        }
    }

    private void WriteCapture(in AttentionCall call, string dir, string shape, int ordinal)
    {
        using OpScope _op = EnterOp();
        _stream.Synchronize();
        int group = call.KvGroup;
        int[] heads = Spread(call.Heads, CaptureHeads), rows = Spread(call.QueryTokens, CaptureQueryRows);
        int[] kvHeads = [.. heads.Select(h => h / group)];
        string target = Path.Combine(dir, $"{shape}-call{ordinal}");
        Directory.CreateDirectory(target);
        WriteRows(call.Query, heads, rows, Path.Combine(target, "q.bin"));
        int[] keys = Spread(call.ValidKvLength, call.ValidKvLength);
        WriteRows(call.Key, kvHeads, keys, Path.Combine(target, "k.bin"));
        WriteRows(call.Value, kvHeads, keys, Path.Combine(target, "v.bin"));
        StringBuilder json = new();
        json.Append(CultureInfo.InvariantCulture,
            $"{{\"shape\":\"{shape}\",\"ordinal\":{ordinal},\"dtype\":\"{call.Query.DType}\",\"batch\":{call.Batch},\"heads\":{call.Heads},");
        json.Append(CultureInfo.InvariantCulture,
            $"\"kvHeads\":{call.KvHeads},\"queryTokens\":{call.QueryTokens},\"keys\":{call.ValidKvLength},\"headDim\":{call.QkHeadDim},");
        json.Append(CultureInfo.InvariantCulture,
            $"\"scale\":{call.Scale:R},\"route\":\"{call.Route}\",\"layout\":\"{call.Query.Layout}\",\"bias\":{(call.Mask.Bias is null ? "false" : "true")},");
        json.Append(CultureInfo.InvariantCulture, $"\"allowF16\":{(call.Numerics.LegacyAssertedBounded ? "true" : "false")},");
        json.Append("\"capturedHeads\":[").AppendJoin(',', heads).Append("],\"capturedKvHeads\":[").AppendJoin(',', kvHeads);
        json.Append("],\"queryRows\":[").AppendJoin(',', rows).Append("],\"files\":{\"q\":\"[heads, queryRows, headDim]\",");
        json.Append("\"k\":\"[heads, keys, headDim]\",\"v\":\"[heads, keys, headDim]\"}}\n");
        File.WriteAllText(Path.Combine(target, "meta.json"), json.ToString());
        Logs.Info($"[attention capture] {shape} call {ordinal} -> {target}");
    }

    /// <summary>Whether <paramref name="list"/> names <paramref name="ordinal"/> for <paramref name="shape"/>: an entry is an ordinal
    /// for every shape, or <c>text@ordinal</c> for the shapes containing <c>text</c>; an entry that is not one is ignored.</summary>
    internal static bool CaptureWanted(string? list, string shape, int ordinal)
    {
        if (list is not { Length: > 0 }) return false;
        foreach (string item in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int at = item.LastIndexOf('@');
            if (at >= 0 && !shape.Contains(item[..at], StringComparison.Ordinal)) continue;
            if (int.TryParse(item[(at + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int wanted) && wanted == ordinal)
                return true;
        }
        return false;
    }

    /// <summary><paramref name="count"/> indices evenly spaced over <c>[0, extent)</c>, first and last included.</summary>
    internal static int[] Spread(int extent, int count)
    {
        int n = Math.Min(extent, count);
        if (n <= 1) return [0];
        return [.. Enumerable.Range(0, n).Select(i => (int)((long)i * (extent - 1) / (n - 1))).Distinct()];
    }

    /// <summary>Batch 0's <paramref name="rows"/> of <paramref name="heads"/> of an operand, in its storage dtype, head-major.</summary>
    private unsafe void WriteRows(in AttentionOperand operand, int[] heads, int[] rows, string path)
    {
        Tensor storage = operand.Storage;
        int elem = storage.DType == DType.F32 ? 4 : storage.DType == DType.F16 || storage.DType == DType.BF16 ? 2
            : throw new NotSupportedException($"Attention capture writes F32, F16 or BF16 operands; got {storage.DType}.");
        long storageBytes = (long)GpuTransferHelper.ByteSize(storage);
        if (storageBytes > Array.MaxLength)
            throw new NotSupportedException($"Attention capture reads whole operands into one array; {storageBytes} bytes is too many.");
        byte[] host = new byte[storageBytes];
        if (GpuTransferHelper.HasCachedActivation(storage))
        {
            ulong device = GpuTransferHelper.CopyToDevice(storage);
            fixed (byte* p = host) CudaDriverApi.cuMemcpyDtoH((nint)p, device, (nuint)storageBytes).ThrowOnError();
        }
        else
        {
            new ReadOnlySpan<byte>((void*)storage.DataPointer, (int)storageBytes).CopyTo(host);
        }
        int rowBytes = operand.HeadDim * elem;
        using FileStream file = File.Create(path);
        foreach (int h in heads)
            foreach (int r in rows)
                file.Write(host.AsSpan(checked((int)((operand.Offset + h * operand.HeadStride + (long)r * operand.TokenStride) * elem)), rowBytes));
    }
}
