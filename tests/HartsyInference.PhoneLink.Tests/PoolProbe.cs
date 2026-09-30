namespace HartsyInference.PhoneLink.Tests;

/// <summary>Asks <see cref="ArrayPool{T}.Shared"/> whether a given array is available for rent. A returned array sits at the
/// top of the calling thread's bucket, so renting from the same thread yields it first; the probe must therefore run on the
/// thread that disposed the owner and before any await.</summary>
internal static class PoolProbe
{
    public static bool IsInPool(byte[] array)
    {
        List<byte[]> rented = new();
        bool found = false;
        for (int i = 0; i < 4 && !found; i++)
        {
            byte[] candidate = ArrayPool<byte>.Shared.Rent(array.Length);
            rented.Add(candidate);
            found = ReferenceEquals(candidate, array);
        }
        foreach (byte[] candidate in rented) ArrayPool<byte>.Shared.Return(candidate);
        return found;
    }

    public static byte[] ArrayOf(ReadOnlyMemory<byte> memory) =>
        MemoryMarshal.TryGetArray(memory, out ArraySegment<byte> segment) && segment.Array is not null
            ? segment.Array
            : throw new InvalidOperationException("Memory is not array-backed.");
}
