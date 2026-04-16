using System.Collections.Concurrent;

namespace SharpInference.Tokenizers;

/// <summary>Thread-safe cache for reusing tokenizer instances across pipeline invocations. Tokenizers are keyed by their model path.</summary>
public sealed class TokenizerCache : IDisposable
{
    private readonly ConcurrentDictionary<string, ClipTokenizer> _clipCache = new();
    private readonly ConcurrentDictionary<string, T5Tokenizer> _t5Cache = new();
    private int _disposed;

    /// <summary>Gets or creates a CLIP tokenizer for the given vocab and merges files.</summary>
    public ClipTokenizer GetOrCreateClip(string vocabPath, string mergesPath)
    {
        ThrowIfDisposed();
        string key = $"{vocabPath}|{mergesPath}";
        return _clipCache.GetOrAdd(key, _ => new ClipTokenizer(vocabPath, mergesPath));
    }

    /// <summary>Gets or creates a T5 tokenizer for the given model file and max length.</summary>
    public T5Tokenizer GetOrCreateT5(string modelPath, int maxLength = T5Tokenizer.DefaultMaxLength)
    {
        ThrowIfDisposed();
        string key = $"{modelPath}|{maxLength}";
        return _t5Cache.GetOrAdd(key, _ => new T5Tokenizer(modelPath, maxLength));
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(TokenizerCache));
    }

    /// <summary>Disposes all cached tokenizer instances.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            foreach (ClipTokenizer tokenizer in _clipCache.Values)
            {
                tokenizer.Dispose();
            }
            _clipCache.Clear();

            foreach (T5Tokenizer tokenizer in _t5Cache.Values)
            {
                tokenizer.Dispose();
            }
            _t5Cache.Clear();
        }
    }
}
