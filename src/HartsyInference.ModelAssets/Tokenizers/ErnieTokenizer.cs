using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>ERNIE-Image (<c>baidu/ERNIE-Image</c>) text-encoder tokenizer — a Mistral3/"ministral3" byte-level BPE shipped as a single HuggingFace-tokenizers <c>tokenizer.json</c> (NOT tekken/tiktoken). The constructor parses <c>model.vocab</c> + <c>model.merges</c> out of the JSON, synthesizes in-memory <c>vocab.json</c>/<c>merges.txt</c> streams, and drives the same <see cref="BpeTokenizer"/> path as <see cref="Qwen3Tokenizer"/>/<see cref="LlamaTokenizer"/>. Special ids follow the Mistral convention: <c>&lt;unk&gt;</c>=0, <c>&lt;s&gt;</c>=1, <c>&lt;/s&gt;</c>=2, <c>&lt;pad&gt;</c>=11.
///
/// <para>Per diffusers' <c>pipeline_ernie_image.py:encode_prompt</c>: NO chat template, raw prompt tokenized with <c>add_special_tokens=True</c> (BOS prepended), NO padding, NO attention mask, truncation at 2048. Whether HF appends EOS for this tokenizer depends on the JSON's post_processor — <see cref="Encode"/> exposes <c>appendEos</c> (default true, validation-gated pending a Python token dump).</para></summary>
public sealed class ErnieTokenizer : IDisposable
{
    /// <summary>Token id for <c>&lt;s&gt;</c>.</summary>
    public const int BosId = 1;

    /// <summary>Token id for <c>&lt;/s&gt;</c>.</summary>
    public const int EosId = 2;

    /// <summary>Token id for <c>&lt;pad&gt;</c>.</summary>
    public const int PadId = 11;

    /// <summary>Token id for <c>&lt;unk&gt;</c>.</summary>
    public const int UnkId = 0;

    /// <summary>Truncation cap — ERNIE-Image's <c>model_max_length</c>.</summary>
    public const int MaxLength = 2048;

    private readonly Tokenizer _tokenizer;
    private readonly int _vocabSize;
    private int _disposed;

    /// <summary>Loads a HuggingFace fast-tokenizer <c>tokenizer.json</c> (top-level <c>model.vocab</c> + <c>model.merges</c>, BPE type).</summary>
    public ErnieTokenizer(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"tokenizer.json not found: {tokenizerJsonPath}", tokenizerJsonPath);

        using FileStream fs = File.OpenRead(tokenizerJsonPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        JsonElement root = doc.RootElement;
        if (!root.TryGetProperty("model", out JsonElement model))
            throw new InvalidDataException($"{tokenizerJsonPath}: missing top-level \"model\" element.");
        if (model.TryGetProperty("type", out JsonElement modelType) && modelType.GetString() != "BPE")
            throw new InvalidDataException($"{tokenizerJsonPath}: model.type is \"{modelType.GetString()}\", expected \"BPE\".");

        using MemoryStream vocabStream = BuildVocabStream(model, out _vocabSize);
        using MemoryStream mergesStream = BuildMergesStream(model);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Number of entries in <c>model.vocab</c> (added/special tokens live outside this count).</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Encodes a raw prompt (no chat template, no padding), truncated to <see cref="MaxLength"/> including specials.</summary>
    /// <param name="addBos">Prepend <see cref="BosId"/> — diffusers tokenizes with <c>add_special_tokens=True</c>, which prepends BOS for Mistral-family tokenizers.</param>
    /// <param name="appendEos">Append <see cref="EosId"/>. Validation-gated default: true.</param>
    public int[] Encode(string text, bool addBos = true, bool appendEos = true)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> bpe = _tokenizer.EncodeToIds(text);

        int specials = (addBos ? 1 : 0) + (appendEos ? 1 : 0);
        int textCount = Math.Min(bpe.Count, MaxLength - specials);
        int[] result = new int[textCount + specials];
        int cursor = 0;
        if (addBos)
            result[cursor++] = BosId;
        for (int i = 0; i < textCount; i++)
            result[cursor++] = bpe[i];
        if (appendEos)
            result[cursor] = EosId;
        return result;
    }

    /// <summary>Encodes raw text with no BOS/EOS/truncation — for round-trip checks against the Python tokenizer.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Serializes <c>model.vocab</c> ({token → id}) back out as a standalone vocab.json stream.</summary>
    private static MemoryStream BuildVocabStream(JsonElement model, out int vocabSize)
    {
        if (!model.TryGetProperty("vocab", out JsonElement vocab))
            throw new InvalidDataException("tokenizer.json: missing model.vocab.");

        MemoryStream stream = new MemoryStream(1 << 22);
        int count = 0;
        using (Utf8JsonWriter writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            foreach (JsonProperty entry in vocab.EnumerateObject())
            {
                writer.WriteNumber(entry.Name, entry.Value.GetInt32());
                count++;
            }
            writer.WriteEndObject();
        }
        vocabSize = count;
        stream.Position = 0;
        return stream;
    }

    /// <summary>Serializes <c>model.merges</c> as a merges.txt stream. Handles both serialization forms the HF tokenizers library emits: legacy <c>"left right"</c> strings and newer <c>["left", "right"]</c> pairs.</summary>
    private static MemoryStream BuildMergesStream(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out JsonElement merges))
            throw new InvalidDataException("tokenizer.json: missing model.merges.");

        MemoryStream stream = new MemoryStream(1 << 22);
        using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false), leaveOpen: true))
        {
            // Same header as the embedded qwen3_merges.txt — BpeTokenizer skips "#version" lines.
            writer.Write("#version: 0.2\n");
            foreach (JsonElement merge in merges.EnumerateArray())
            {
                if (merge.ValueKind == JsonValueKind.String)
                {
                    writer.Write(merge.GetString());
                }
                else
                {
                    int part = 0;
                    foreach (JsonElement piece in merge.EnumerateArray())
                    {
                        if (part++ > 0) writer.Write(' ');
                        writer.Write(piece.GetString());
                    }
                }
                writer.Write('\n');
            }
        }
        stream.Position = 0;
        return stream;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Microsoft.ML.Tokenizers.Tokenizer is not IDisposable in v2.0; nothing to free.
        }
    }
}
