namespace SharpInference.Core.Backends;

/// <summary>Opaque single-use handle to an in-flight weight upload. Pass to <see cref="IStreamingWeightCache.AwaitWeights"/> exactly once; the cache disposes the underlying completion handle on await.</summary>
public readonly struct StreamingUploadToken : IEquatable<StreamingUploadToken>
{
    /// <summary>The backend-specific completion handle (e.g., CUevent on CUDA).</summary>
    public nint Handle { get; }

    /// <summary>Identifies which backend issued this token; the cache verifies the issuer on await.</summary>
    public object BackendTag { get; }

    public StreamingUploadToken(nint handle, object backendTag)
    {
        Handle = handle;
        BackendTag = backendTag;
    }

    /// <summary>Sentinel no-op token returned when there's nothing to wait for (e.g. all tensors were already cached). <see cref="IStreamingWeightCache.AwaitWeights"/> short-circuits on this value.</summary>
    public static StreamingUploadToken Empty => default;

    public bool IsEmpty => Handle == 0 && BackendTag is null;

    public bool Equals(StreamingUploadToken other) =>
        Handle == other.Handle && ReferenceEquals(BackendTag, other.BackendTag);

    public override bool Equals(object? obj) => obj is StreamingUploadToken t && Equals(t);
    public override int GetHashCode() => HashCode.Combine(Handle, BackendTag);
    public static bool operator ==(StreamingUploadToken a, StreamingUploadToken b) => a.Equals(b);
    public static bool operator !=(StreamingUploadToken a, StreamingUploadToken b) => !a.Equals(b);
}
