namespace SharpInference.Core.Backends;

/// <summary>
/// Opaque handle to an in-flight weight upload, returned from
/// <see cref="IStreamingWeightCache.BeginUploadAsync"/>. Wraps a backend-specific
/// completion handle (e.g., a CUDA event). Single-use — pass to
/// <see cref="IStreamingWeightCache.AwaitWeights"/> exactly once; the cache disposes
/// the underlying handle on await.
///
/// <para>The struct is intentionally tiny (one <c>nint</c>) so it can be stored in
/// per-block state without pressure on the GC. The <c>BackendTag</c> is a sanity
/// guard against passing a token to the wrong backend's cache.</para>
/// </summary>
public readonly struct StreamingUploadToken : IEquatable<StreamingUploadToken>
{
    /// <summary>The backend-specific completion handle (e.g., CUevent on CUDA).</summary>
    public nint Handle { get; }

    /// <summary>Identifies which backend issued this token. The cache verifies on
    /// <see cref="IStreamingWeightCache.AwaitWeights"/> that it was the issuer.</summary>
    public object BackendTag { get; }

    public StreamingUploadToken(nint handle, object backendTag)
    {
        Handle = handle;
        BackendTag = backendTag;
    }

    /// <summary>A sentinel "no-op" token returned when there's nothing to wait for
    /// (e.g., all tensors were already cached). <see cref="IStreamingWeightCache.AwaitWeights"/>
    /// short-circuits on this value.</summary>
    public static StreamingUploadToken Empty => default;

    public bool IsEmpty => Handle == 0 && BackendTag is null;

    public bool Equals(StreamingUploadToken other) =>
        Handle == other.Handle && ReferenceEquals(BackendTag, other.BackendTag);

    public override bool Equals(object? obj) => obj is StreamingUploadToken t && Equals(t);
    public override int GetHashCode() => HashCode.Combine(Handle, BackendTag);
    public static bool operator ==(StreamingUploadToken a, StreamingUploadToken b) => a.Equals(b);
    public static bool operator !=(StreamingUploadToken a, StreamingUploadToken b) => !a.Equals(b);
}
