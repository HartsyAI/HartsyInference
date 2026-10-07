using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Audio.Frontends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Audio;

/// <summary>Installs what Kokoro's Japanese front-end reads into the shared audio folder: full UniDic 3.1.0
/// (2021-08-31, BSD/LGPL/GPL), the MeCab dictionary the <c>unidic</c> Python package downloads for fugashi under
/// misaki, and misaki's <c>ja_words.txt</c> (Apache-2.0) at the commit the English dictionaries are pinned to. The
/// UniDic archive is streamed once: its SHA-256 is checked while only the files MeCab needs are unpacked, so the
/// archive itself never lands on disk.</summary>
internal static class KokoroJapaneseAssets
{
    private const string UniDicUrl = "https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip";
    private const string UniDicSha256 = "638718c4c63625ab300de4c92c67925d54c0e9e3830009eaa992f29819d59c43";
    private const long UniDicBytes = 525_943_040;
    private const string WordsUrl = "https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3/misaki/data/ja_words.txt";
    private const string WordsSha256 = "a93a8e8aee24db307a32becb8bf01c4c2908ecf37e6c91f7a705fafdfeba67ff";
    private const string ArchivePrefix = "unidic/";
    private const uint LocalHeaderSignature = 0x04034b50;

    // Archive member (under unidic/) to installed file name; the licenses travel with the data.
    private static readonly Dictionary<string, string> Members = new(StringComparer.Ordinal)
    {
        ["sys.dic"] = "sys.dic", ["unk.dic"] = "unk.dic", ["matrix.bin"] = "matrix.bin", ["char.bin"] = "char.bin",
        ["dicrc"] = "dicrc", ["licenses/COPYING"] = "COPYING", ["licenses/BSD"] = "BSD",
    };

    private static readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Where the dictionary and word list live.</summary>
    public static string Directory => AudioModelRoot.SharedFile("unidic-3.1.0");

    /// <summary>Installs the dictionary and word list unless present, and returns their directory.</summary>
    public static async Task<string> EnsureAsync(CancellationToken cancel)
    {
        string target = Directory;
        if (IsInstalled(target)) return target;
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            if (!File.Exists(Path.Combine(target, "sys.dic"))) await InstallUniDicAsync(target, cancel).ConfigureAwait(false);
            string words = Path.Combine(target, KokoroJapaneseG2P.WordsFileName);
            if (!File.Exists(words))
            {
                Logs.Info("[Audio][Kokoro] Downloading misaki's Japanese word list (ja_words.txt, ~2 MB)...");
                await AudioFileFetcher.EnsureAsync(WordsUrl, words, WordsSha256, cancel).ConfigureAwait(false);
            }
            return target;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Installs the assets if needed and opens the Japanese front-end over them.</summary>
    public static async Task<KokoroJapaneseG2P> LoadAsync(CancellationToken cancel)
    {
        string directory = await EnsureAsync(cancel).ConfigureAwait(false);
        return new KokoroJapaneseG2P(directory);
    }

    private static bool IsInstalled(string directory) =>
        File.Exists(Path.Combine(directory, "sys.dic")) && File.Exists(Path.Combine(directory, KokoroJapaneseG2P.WordsFileName));

    private static async Task InstallUniDicAsync(string target, CancellationToken cancel)
    {
        // Staged under a name unique to this call, so engines sharing one cache never write into each other's files.
        string staging = $"{target}.{Guid.NewGuid():N}.tmp";
        try
        {
            System.IO.Directory.CreateDirectory(staging);
            Logs.Info($"[Audio][Kokoro] Downloading UniDic 3.1.0 for Japanese (one-time, {ByteFormat.Mb(UniDicBytes)} "
                + "download, ~690 MB installed)...");
            using HttpClient client = new() { Timeout = Timeout.InfiniteTimeSpan };
            using HttpResponseMessage response = await client.GetAsync(UniDicUrl, HttpCompletionOption.ResponseHeadersRead, cancel)
                .ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            await using Stream body = await response.Content.ReadAsStreamAsync(cancel).ConfigureAwait(false);
            using HashingStream hashed = new(body);
            HashSet<string> found = await ExtractAsync(hashed, staging, cancel).ConfigureAwait(false);
            string actual = hashed.FinishHex();
            if (!string.Equals(actual, UniDicSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"'{UniDicUrl}' has SHA-256 {actual}, expected {UniDicSha256}; discarded.");
            if (found.Count != Members.Count)
                throw new InvalidDataException($"'{UniDicUrl}' lacks {string.Join(", ", Members.Keys.Except(found))}.");
            // Each file is published with one rename, sys.dic last: it is what marks the dictionary installed.
            System.IO.Directory.CreateDirectory(target);
            foreach (string file in System.IO.Directory.EnumerateFiles(staging).OrderBy(f => Path.GetFileName(f) == "sys.dic"))
                File.Move(file, Path.Combine(target, Path.GetFileName(file)), overwrite: true);
            Logs.Info($"[Audio][Kokoro] UniDic installed at '{target}'.");
        }
        finally
        {
            if (System.IO.Directory.Exists(staging)) System.IO.Directory.Delete(staging, recursive: true);
        }
    }

    // Walks the zip's local file headers front to back (sizes are in each header: the archive sets no data
    // descriptors), unpacking the wanted members and reading past the rest so the whole archive is hashed.
    private static async Task<HashSet<string>> ExtractAsync(Stream zip, string directory, CancellationToken cancel)
    {
        HashSet<string> found = new(StringComparer.Ordinal);
        byte[] header = new byte[30];
        while (await ReadFullyAsync(zip, header.AsMemory(0, 4), cancel).ConfigureAwait(false)
            && BinaryPrimitives.ReadUInt32LittleEndian(header) == LocalHeaderSignature)
        {
            if (!await ReadFullyAsync(zip, header.AsMemory(4, 26), cancel).ConfigureAwait(false))
                throw new InvalidDataException("The UniDic archive ends inside a file header.");
            int flags = BinaryPrimitives.ReadUInt16LittleEndian(header.AsSpan(6));
            int method = BinaryPrimitives.ReadUInt16LittleEndian(header.AsSpan(8));
            long compressed = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(18));
            long size = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(22));
            byte[] name = new byte[BinaryPrimitives.ReadUInt16LittleEndian(header.AsSpan(26))];
            byte[] extra = new byte[BinaryPrimitives.ReadUInt16LittleEndian(header.AsSpan(28))];
            if (!await ReadFullyAsync(zip, name, cancel).ConfigureAwait(false)
                || !await ReadFullyAsync(zip, extra, cancel).ConfigureAwait(false))
                throw new InvalidDataException("The UniDic archive ends inside a file header.");
            if (compressed == 0xFFFFFFFF || size == 0xFFFFFFFF)
                throw new InvalidDataException("The UniDic archive uses Zip64 sizes; this reader handles the pinned archive's plain zip only.");
            if ((flags & 8) != 0) throw new InvalidDataException("The UniDic archive uses data descriptors; cannot stream it.");
            string entry = Encoding.UTF8.GetString(name);
            await using BoundedStream data = new(zip, compressed);
            if (entry.StartsWith(ArchivePrefix, StringComparison.Ordinal)
                && Members.TryGetValue(entry[ArchivePrefix.Length..], out string? fileName))
            {
                if (method is not (0 or 8)) throw new InvalidDataException($"UniDic member '{entry}' uses zip method {method}.");
                string path = Path.Combine(directory, fileName);
                long written;
                await using (FileStream file = File.Create(path))
                {
                    if (method == 8)
                    {
                        await using DeflateStream inflate = new(data, CompressionMode.Decompress, leaveOpen: true);
                        await inflate.CopyToAsync(file, 1 << 20, cancel).ConfigureAwait(false);
                    }
                    else
                    {
                        await data.CopyToAsync(file, 1 << 20, cancel).ConfigureAwait(false);
                    }
                    written = file.Length;
                }
                if (written != size) throw new InvalidDataException($"UniDic member '{entry}' unpacked to {written} bytes, expected {size}.");
                found.Add(entry[ArchivePrefix.Length..]);
            }
            await data.DrainAsync(cancel).ConfigureAwait(false);
        }
        await zip.CopyToAsync(Stream.Null, 1 << 20, cancel).ConfigureAwait(false);
        return found;
    }

    private static async Task<bool> ReadFullyAsync(Stream stream, Memory<byte> buffer, CancellationToken cancel)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int read = await stream.ReadAsync(buffer[total..], cancel).ConfigureAwait(false);
            if (read == 0) return false;
            total += read;
        }
        return true;
    }

    // Forward-only view that hashes everything read through it.
    private sealed class HashingStream(Stream inner) : Stream
    {
        private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public string FinishHex() => Convert.ToHexString(_hash.GetHashAndReset()).ToLowerInvariant();

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            int read = inner.Read(buffer);
            _hash.AppendData(buffer[..read]);
            return read;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            int read = await inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
            _hash.AppendData(buffer.Span[..read]);
            return read;
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing) _hash.Dispose();
            base.Dispose(disposing);
        }
    }

    // At most `length` bytes of the inner stream, so inflation cannot read into the next member; the inner stream
    // stays open.
    private sealed class BoundedStream(Stream inner, long length) : Stream
    {
        private long _remaining = length;

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public async Task DrainAsync(CancellationToken cancel)
        {
            byte[] scratch = new byte[1 << 16];
            while (_remaining > 0)
            {
                if (await ReadAsync(scratch, cancel).ConfigureAwait(false) == 0)
                    throw new InvalidDataException("The UniDic archive ends inside a member.");
            }
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            if (_remaining == 0) return 0;
            int read = inner.Read(buffer[..(int)Math.Min(buffer.Length, _remaining)]);
            _remaining -= read;
            return read;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (_remaining == 0) return 0;
            int read = await inner.ReadAsync(buffer[..(int)Math.Min(buffer.Length, _remaining)], cancellationToken)
                .ConfigureAwait(false);
            _remaining -= read;
            return read;
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
