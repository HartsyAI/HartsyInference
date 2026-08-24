using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Validated geometry shared by the host and accelerator patch-token shuffles.</summary>
internal readonly record struct PatchTokenGeometry(int Batch, int Channels, int Height, int Width, int HPacked,
    int WPacked, long SequenceLength, long PatchVolume, long ElementCount);

/// <summary>Centralizes the exact tensor contract for <see cref="IBackend.PatchifyTokens"/> and <see cref="IBackend.UnpatchifyTokens"/> so CPU fallbacks and accelerator dispatch cannot silently disagree.</summary>
internal static class PatchTokenContract
{
    internal static PatchTokenGeometry ValidatePatchify(Tensor output, Tensor input, int patch)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ValidateCommon(output, input, patch, "PatchifyTokens");

        if (input.Shape.Rank != 4)
            throw new ArgumentException(
                $"PatchifyTokens input must be rank-4 [B,C,H,W]; got {input.Shape}.", nameof(input));
        if (output.Shape.Rank != 3)
            throw new ArgumentException(
                $"PatchifyTokens output must be rank-3 [B,(H/p)*(W/p),p*p*C]; got {output.Shape}.", nameof(output));

        int batch = PositiveDimension(input.Shape[0], "B", nameof(input));
        int channels = PositiveDimension(input.Shape[1], "C", nameof(input));
        int height = PositiveDimension(input.Shape[2], "H", nameof(input));
        int width = PositiveDimension(input.Shape[3], "W", nameof(input));
        if (height % patch != 0 || width % patch != 0)
            throw new ArgumentException(
                $"PatchifyTokens requires H and W divisible by patch={patch}; got input {input.Shape}.", nameof(input));

        PatchTokenGeometry geometry = ValidateShapes(
            output, input, batch, channels, height, width, patch, patchify: true);
        ValidateNoStorageOverlap(output, input, "PatchifyTokens");
        return geometry;
    }

    internal static PatchTokenGeometry ValidateUnpatchify(
        Tensor output, Tensor tokens, int channels, int hPacked, int wPacked, int patch)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(tokens);
        ValidateCommon(output, tokens, patch, "UnpatchifyTokens");

        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "UnpatchifyTokens channels must be positive.");
        if (hPacked <= 0)
            throw new ArgumentOutOfRangeException(nameof(hPacked), hPacked, "UnpatchifyTokens hPacked must be positive.");
        if (wPacked <= 0)
            throw new ArgumentOutOfRangeException(nameof(wPacked), wPacked, "UnpatchifyTokens wPacked must be positive.");
        if (tokens.Shape.Rank != 3)
            throw new ArgumentException(
                $"UnpatchifyTokens tokens must be rank-3 [B,hPacked*wPacked,C*p*p]; got {tokens.Shape}.", nameof(tokens));
        if (output.Shape.Rank != 4)
            throw new ArgumentException(
                $"UnpatchifyTokens output must be rank-4 [B,C,hPacked*p,wPacked*p]; got {output.Shape}.", nameof(output));

        int batch = PositiveDimension(tokens.Shape[0], "B", nameof(tokens));
        int height;
        int width;
        try
        {
            height = checked(hPacked * patch);
            width = checked(wPacked * patch);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(nameof(patch), patch,
                "UnpatchifyTokens spatial geometry exceeds the signed 32-bit indexing contract.");
        }

        PatchTokenGeometry geometry = ValidateShapes(
            output, tokens, batch, channels, height, width, patch, patchify: false);
        ValidateNoStorageOverlap(output, tokens, "UnpatchifyTokens");
        return geometry;
    }

    private static void ValidateCommon(Tensor output, Tensor input, int patch, string operation)
    {
        if (ReferenceEquals(output, input))
            throw new ArgumentException($"{operation} output must not alias its input.", nameof(output));
        if (patch <= 0)
            throw new ArgumentOutOfRangeException(nameof(patch), patch, $"{operation} patch must be positive.");
        if (output.DType != input.DType)
            throw new ArgumentException(
                $"{operation} requires matching dtypes; got output={output.DType}, input={input.DType}.");
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException(
                $"{operation} supports the byte-addressable F32, F16, and BF16 dtypes; got {input.DType}.");
    }

    private static void ValidateNoStorageOverlap(Tensor output, Tensor input, string operation)
    {
        // Shape/dtype/byte-span validation has completed before this storage-range query. The helper never
        // materializes lazy host storage, so contract checking cannot turn a resident CUDA shuffle into a D2H.
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException(
                $"{operation} output storage must not overlap its input storage.", nameof(output));
    }

    private static PatchTokenGeometry ValidateShapes(
        Tensor output,
        Tensor input,
        int batch,
        int channels,
        int height,
        int width,
        int patch,
        bool patchify)
    {
        int hPacked = height / patch;
        int wPacked = width / patch;
        long sequenceLength;
        long patchVolume;
        long elementCount;
        try
        {
            sequenceLength = checked((long)hPacked * wPacked);
            patchVolume = checked((long)channels * patch * patch);
            elementCount = checked((long)batch * channels * height * width);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(nameof(input),
                $"{(patchify ? "PatchifyTokens" : "UnpatchifyTokens")} geometry exceeds 64-bit tensor indexing.");
        }
        if (elementCount > long.MaxValue / input.DType.SizeInBytes)
            throw new ArgumentOutOfRangeException(nameof(input),
                $"{(patchify ? "PatchifyTokens" : "UnpatchifyTokens")} byte span exceeds signed 64-bit addressable storage.");

        TensorShape expectedInput = patchify ? new TensorShape(batch, channels, height, width)
            : new TensorShape(batch, sequenceLength, patchVolume);
        TensorShape expectedOutput = patchify ? new TensorShape(batch, sequenceLength, patchVolume)
            : new TensorShape(batch, channels, height, width);

        if (input.Shape != expectedInput)
            throw new ArgumentException(
                $"{(patchify ? "PatchifyTokens" : "UnpatchifyTokens")} input shape must be {expectedInput}; got {input.Shape}.",
                nameof(input));
        if (output.Shape != expectedOutput)
            throw new ArgumentException(
                $"{(patchify ? "PatchifyTokens" : "UnpatchifyTokens")} output shape must be {expectedOutput}; got {output.Shape}.",
                nameof(output));
        if (input.ElementCount != elementCount || output.ElementCount != elementCount)
            throw new ArgumentException(
                $"{(patchify ? "PatchifyTokens" : "UnpatchifyTokens")} tensors must each contain exactly {elementCount} elements.");

        return new PatchTokenGeometry(
            batch, channels, height, width, hPacked, wPacked, sequenceLength, patchVolume, elementCount);
    }

    private static int PositiveDimension(long value, string axis, string parameterName)
    {
        if (value <= 0 || value > int.MaxValue)
            throw new ArgumentOutOfRangeException(
                parameterName, value, $"Patch-token axis {axis} must be in [1,{int.MaxValue}].");
        return (int)value;
    }
}

/// <summary>Byte-exact host reference for patch-token permutations.</summary>
internal static unsafe class PatchTokenHostShuffle
{
    internal static void Patchify(
        Tensor output, Tensor input, PatchTokenGeometry geometry, int patch, bool innerChannelFastest)
    {
        if (input.DType == DType.F32)
            PatchifyCore<uint>(output, input, geometry, patch, innerChannelFastest);
        else
            PatchifyCore<ushort>(output, input, geometry, patch, innerChannelFastest);
    }

    internal static void Unpatchify(
        Tensor output, Tensor input, PatchTokenGeometry geometry, int patch, bool innerChannelFastest)
    {
        if (input.DType == DType.F32)
            UnpatchifyCore<uint>(output, input, geometry, patch, innerChannelFastest);
        else
            UnpatchifyCore<ushort>(output, input, geometry, patch, innerChannelFastest);
    }

    private static void PatchifyCore<T>(
        Tensor output, Tensor input, PatchTokenGeometry geometry, int patch, bool innerChannelFastest)
        where T : unmanaged
    {
        T* src = (T*)input.DataPointer;
        T* dst = (T*)output.DataPointer;
        long hw = (long)geometry.Height * geometry.Width;
        for (int b = 0; b < geometry.Batch; b++)
            for (int hp = 0; hp < geometry.HPacked; hp++)
                for (int wp = 0; wp < geometry.WPacked; wp++)
                    for (int ph = 0; ph < patch; ph++)
                        for (int pw = 0; pw < patch; pw++)
                            for (int c = 0; c < geometry.Channels; c++)
                            {
                                int y = hp * patch + ph;
                                int x = wp * patch + pw;
                                long seq = (long)hp * geometry.WPacked + wp;
                                long inner = innerChannelFastest ? (((long)ph * patch + pw) * geometry.Channels + c)
                                    : (((long)c * patch + ph) * patch + pw);
                                long srcIndex = ((long)b * geometry.Channels + c) * hw
                                    + (long)y * geometry.Width + x;
                                long dstIndex = ((long)b * geometry.SequenceLength + seq) * geometry.PatchVolume + inner;
                                dst[dstIndex] = src[srcIndex];
                            }

        GC.KeepAlive(input);
        GC.KeepAlive(output);
    }

    private static void UnpatchifyCore<T>(
        Tensor output, Tensor input, PatchTokenGeometry geometry, int patch, bool innerChannelFastest)
        where T : unmanaged
    {
        T* src = (T*)input.DataPointer;
        T* dst = (T*)output.DataPointer;
        long hw = (long)geometry.Height * geometry.Width;
        for (int b = 0; b < geometry.Batch; b++)
            for (int c = 0; c < geometry.Channels; c++)
                for (int y = 0; y < geometry.Height; y++)
                    for (int x = 0; x < geometry.Width; x++)
                    {
                        int hp = y / patch, ph = y % patch;
                        int wp = x / patch, pw = x % patch;
                        long seq = (long)hp * geometry.WPacked + wp;
                        long inner = innerChannelFastest ? (((long)ph * patch + pw) * geometry.Channels + c)
                            : (((long)c * patch + ph) * patch + pw);
                        long dstIndex = ((long)b * geometry.Channels + c) * hw
                            + (long)y * geometry.Width + x;
                        long srcIndex = ((long)b * geometry.SequenceLength + seq) * geometry.PatchVolume + inner;
                        dst[dstIndex] = src[srcIndex];
                    }

        GC.KeepAlive(input);
        GC.KeepAlive(output);
    }
}
