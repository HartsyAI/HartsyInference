using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Latent-cache attention, indexer, cache quantization, window and offset-rope primitives. The latent shaders read byte
// caches as 32-bit words, so a byte tensor whose size is not a multiple of four is staged through a padded temporary.
public sealed partial class VulkanBackend
{
    /// <summary>Largest k the attention kernel keeps as probabilities in shared memory (the 16 KB every device has).</summary>
    private const int LatentAttentionMaxK = 4096;

    /// <inheritdoc/>
    public void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale)
    {
        SparseLatentAttentionReference.Validate(output, query, window, main, indices, windowSlots, sink);
        int k = (int)indices.Shape[1];
        if (k > LatentAttentionMaxK)
        {
            throw new NotSupportedException(
                $"Vulkan SparseLatentAttention holds k <= {LatentAttentionMaxK} probabilities in shared memory; got {k}.");
        }
        using OpScope _op = EnterOp();
        long pairs = query.Shape[0] * query.Shape[1];
        List<VulkanBuffer> temps = new();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            BindSource(window, temps, out VulkanBuffer wCodes, out VulkanBuffer wScales);
            BindSource(main, temps, out VulkanBuffer mCodes, out VulkanBuffer mScales);
            VulkanBuffer queryBuf = GetBuffer(query);
            VulkanBuffer idxBuf = GetBuffer(indices);
            VulkanBuffer sinkBuf = GetBuffer(sink);
            VulkanKernel kernel = GetKernel("sparse_latent_attention", storageBufferCount: 8, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[]
            {
                outBuf.Handle, queryBuf.Handle, wCodes.Handle, wScales.Handle, mCodes.Handle, mScales.Handle,
                idxBuf.Handle, sinkBuf.Handle,
            };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < pairs; done += MaxWorkgroupsX)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)pairs);
                pc.U32((uint)query.Shape[1]);
                pc.U32((uint)query.Shape[2]);
                pc.U32((uint)k);
                pc.U32((uint)windowSlots);
                pc.U32((uint)main.Rows);
                pc.U32((uint)window.Encoding);
                pc.U32((uint)main.Encoding);
                pc.F32(scale);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, (uint)Math.Min(pairs - done, MaxWorkgroupsX));
            }
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SparseLatentAttention dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }

    /// <inheritdoc/>
    public void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates, float scale)
    {
        IndexerScoresReference.Validate(scores, query, keys, headWeights, compressLens, candidates);
        using OpScope _op = EnterOp();
        long total = scores.ElementCount;
        if (total == 0)
        {
            return;
        }
        List<VulkanBuffer> temps = new();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * sizeof(float)));
        try
        {
            BindSource(keys, temps, out VulkanBuffer kCodes, out VulkanBuffer kScales);
            VulkanBuffer queryBuf = GetBuffer(query);
            VulkanBuffer weightBuf = GetBuffer(headWeights);
            VulkanBuffer lenBuf = GetBuffer(compressLens);
            VulkanBuffer candBuf = candidates is null ? Placeholder(temps) : PaddedInput(candidates, temps);
            VulkanKernel kernel = GetKernel("indexer_scores", storageBufferCount: 7, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[]
            {
                outBuf.Handle, queryBuf.Handle, kCodes.Handle, kScales.Handle, weightBuf.Handle, lenBuf.Handle, candBuf.Handle,
            };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)total);
                pc.U32((uint)keys.Rows);
                pc.U32((uint)query.Shape[1]);
                pc.U32((uint)query.Shape[2]);
                pc.U32((uint)keys.Encoding);
                pc.U32(candidates is null ? 0u : 1u);
                pc.F32(scale);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(scores, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan IndexerScores dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }

    /// <inheritdoc/>
    /// <remarks>A destination row past the last row is skipped on the device; the CPU reference throws. The destination
    /// buffers are updated in place, so their byte sizes must be whole 32-bit words.</remarks>
    public void QuantizeLatentRows(in LatentSource dest, Tensor rows, Tensor physicalRows)
    {
        LatentQuantReference.ValidateQuantize(dest, rows, physicalRows);
        RequireWordAligned(dest.Codes!, "QuantizeLatentRows destination codes");
        if (dest.Scales is not null)
        {
            RequireWordAligned(dest.Scales, "QuantizeLatentRows destination scales");
        }
        using OpScope _op = EnterOp();
        long count = physicalRows.ElementCount;
        List<VulkanBuffer> temps = new();
        try
        {
            VulkanBuffer codesBuf = GetBuffer(dest.Codes!);
            VulkanBuffer scalesBuf = dest.Scales is null ? Placeholder(temps) : GetBuffer(dest.Scales);
            VulkanBuffer srcBuf = GetBuffer(rows);
            VulkanBuffer physBuf = GetBuffer(physicalRows);
            VulkanKernel kernel = GetKernel("quantize_latent_rows", storageBufferCount: 4, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { codesBuf.Handle, scalesBuf.Handle, srcBuf.Handle, physBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < count; done += MaxWorkgroupsX)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)count);
                pc.U32((uint)dest.Dim);
                pc.U32((uint)dest.Rows);
                pc.U32((uint)dest.Encoding);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, (uint)Math.Min(count - done, MaxWorkgroupsX));
            }
            CacheOutput(dest.Codes!, codesBuf);
            if (dest.Scales is not null)
            {
                CacheOutput(dest.Scales, scalesBuf);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan QuantizeLatentRows dispatch failed", ex);
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }

    /// <inheritdoc/>
    public void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding)
    {
        LatentQuantReference.ValidateActQuant(x, encoding);
        int group = LatentEncodings.GroupSize(encoding);
        if (group == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        long groups = x.ElementCount / group;
        VulkanBuffer xBuf = GetBuffer(x);
        try
        {
            VulkanKernel kernel = GetKernel("act_quant_dequant", storageBufferCount: 1, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < groups; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)groups);
                pc.U32((uint)encoding);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(groups - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(x, xBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ActQuantDequantInPlace dispatch failed", ex);
            throw;
        }
    }

    /// <inheritdoc/>
    public void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos)
    {
        WindowIndicesReference.Validate(indices, windowSize, seqLen, startPos);
        using OpScope _op = EnterOp();
        (int rows, int cols) = WindowIndicesReference.Shape(windowSize, seqLen, startPos);
        long total = (long)rows * cols;
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * sizeof(int)));
        try
        {
            VulkanKernel kernel = GetKernel("build_window_indices", storageBufferCount: 1, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)total);
            pc.U32((uint)cols);
            pc.U32((uint)windowSize);
            pc.U32((uint)startPos);
            Dispatch(kernel, bufs, pc.Written, GroupCount(total, LocalX1D));
            CacheOutput(indices, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan BuildWindowIndices dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset)
    {
        RopeInterleavedOffsetReference.Validate(x, cos, sin, rotaryDim, dimOffset);
        using OpScope _op = EnterOp();
        int rank = x.Shape.Rank;
        long heads = rank == 4 ? x.Shape[2] : 1;
        long half = rotaryDim / 2;
        long total = x.Shape[0] * x.Shape[1] * heads * half;
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer cosBuf = GetBuffer(cos);
        VulkanBuffer sinBuf = GetBuffer(sin);
        try
        {
            VulkanKernel kernel = GetKernel("rope_interleaved_offset", storageBufferCount: 3, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle, cosBuf.Handle, sinBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)total);
                pc.U32((uint)heads);
                pc.U32((uint)x.Shape[rank - 1]);
                pc.U32((uint)half);
                pc.U32((uint)dimOffset);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(x, xBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ApplyRopeInterleaved (offset) dispatch failed", ex);
            throw;
        }
    }

    private static void RequireWordAligned(Tensor t, string what)
    {
        if (t.ElementCount * t.DType.SizeInBytes % 4 != 0)
        {
            throw new NotSupportedException($"Vulkan {what} must be a whole number of 32-bit words; got {t.ElementCount * t.DType.SizeInBytes} bytes.");
        }
    }

    // A source's code and scale buffers; an empty source, and F32's missing scales, bind an unread placeholder.
    private void BindSource(in LatentSource source, List<VulkanBuffer> temps, out VulkanBuffer codes, out VulkanBuffer scales)
    {
        bool empty = source.Rows == 0 || source.Codes is null;
        codes = empty ? Placeholder(temps) : PaddedInput(source.Codes!, temps);
        scales = empty || source.Scales is null ? Placeholder(temps) : PaddedInput(source.Scales, temps);
    }

    // A read-only input as 32-bit words: the tensor's own buffer when it already is whole words, else a padded copy.
    private VulkanBuffer PaddedInput(Tensor t, List<VulkanBuffer> temps)
    {
        long bytes = t.ElementCount * t.DType.SizeInBytes;
        if (bytes % 4 == 0)
        {
            return GetBuffer(t);
        }
        VulkanBuffer padded = _xfer.AllocateDevice((ulong)((bytes + 3) & ~3L));
        temps.Add(padded);
        _xfer.Upload(padded, t);
        return padded;
    }

    private VulkanBuffer Placeholder(List<VulkanBuffer> temps)
    {
        VulkanBuffer buffer = _xfer.AllocateDevice(4);
        temps.Add(buffer);
        return buffer;
    }

    private void FreeTemps(List<VulkanBuffer> temps)
    {
        foreach (VulkanBuffer buffer in temps)
        {
            _xfer.FreeDevice(buffer);
        }
    }
}
