using System.Buffers.Binary;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>g2p_en's neural letter-to-sound model (<c>G2p.predict</c>): a 64-dim embedding + 128-unit GRU encoder over
/// the word's letters and a GRU decoder that greedily emits up to 20 ARPAbet phones, with the pretrained
/// <c>checkpoint20.npz</c> weights. Spells out-of-vocabulary words for frontends that follow g2p_en.</summary>
internal sealed class G2pEnLetterToSound
{
    private const int MaxSteps = 20;
    private const int StartToken = 2;
    private const int EndToken = 3;

    private static readonly Lazy<G2pEnLetterToSound> Shared = new(Load);

    private readonly Dictionary<char, int> _graphemeIndex;
    private readonly string[] _phonemes;
    private readonly int _unknownGrapheme;
    private readonly int _endGrapheme;
    private readonly Matrix _encEmb, _encWih, _encWhh, _decEmb, _decWih, _decWhh, _fcW;
    private readonly float[] _encBih, _encBhh, _decBih, _decBhh, _fcB;

    private G2pEnLetterToSound(string[] graphemes, string[] phonemes, Dictionary<string, Matrix> t)
    {
        _graphemeIndex = [];
        for (int i = 0; i < graphemes.Length; i++)
            if (graphemes[i].Length == 1) _graphemeIndex[graphemes[i][0]] = i;
        _unknownGrapheme = Array.IndexOf(graphemes, "<unk>");
        _endGrapheme = Array.IndexOf(graphemes, "</s>");
        _phonemes = phonemes;
        _encEmb = t["enc_emb"];
        _encWih = t["enc_w_ih"];
        _encWhh = t["enc_w_hh"];
        _encBih = t["enc_b_ih"].Data;
        _encBhh = t["enc_b_hh"].Data;
        _decEmb = t["dec_emb"];
        _decWih = t["dec_w_ih"];
        _decWhh = t["dec_w_hh"];
        _decBih = t["dec_b_ih"].Data;
        _decBhh = t["dec_b_hh"].Data;
        _fcW = t["fc_w"];
        _fcB = t["fc_b"].Data;
    }

    /// <summary>The model over the embedded g2p_en weights (loaded on first use).</summary>
    public static G2pEnLetterToSound English => Shared.Value;

    /// <summary><c>G2p.predict(word)</c>: ARPAbet phones for a lower-case word; characters outside a-z read as
    /// <c>&lt;unk&gt;</c>.</summary>
    public List<string> Predict(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        int hidden = _encWhh.Cols;
        float[] h = new float[hidden];
        for (int t = 0; t <= word.Length; t++)
        {
            int g = t == word.Length ? _endGrapheme : _graphemeIndex.GetValueOrDefault(word[t], _unknownGrapheme);
            h = GruCell(_encEmb.Row(g), h, _encWih, _encWhh, _encBih, _encBhh);
        }

        List<string> phones = [];
        ReadOnlySpan<float> input = _decEmb.Row(StartToken);
        float[] logits = new float[_fcW.Rows];
        for (int step = 0; step < MaxSteps; step++)
        {
            h = GruCell(input, h, _decWih, _decWhh, _decBih, _decBhh);
            for (int o = 0; o < logits.Length; o++) logits[o] = Dot(_fcW.Row(o), h) + _fcB[o];
            int pred = 0;
            for (int o = 1; o < logits.Length; o++)
                if (logits[o] > logits[pred]) pred = o;     // numpy argmax: first maximum
            if (pred == EndToken) break;
            phones.Add(pred < _phonemes.Length ? _phonemes[pred] : "<unk>");
            input = _decEmb.Row(pred);
        }
        return phones;
    }

    private static float[] GruCell(ReadOnlySpan<float> x, float[] h, Matrix wIh, Matrix wHh, float[] bIh, float[] bHh)
    {
        int n = wHh.Cols;
        float[] next = new float[n];
        for (int j = 0; j < n; j++)
        {
            float r = Sigmoid(Dot(wIh.Row(j), x) + bIh[j] + (Dot(wHh.Row(j), h) + bHh[j]));
            float z = Sigmoid(Dot(wIh.Row(n + j), x) + bIh[n + j] + (Dot(wHh.Row(n + j), h) + bHh[n + j]));
            float nIh = Dot(wIh.Row(2 * n + j), x) + bIh[2 * n + j];
            float nHh = Dot(wHh.Row(2 * n + j), h) + bHh[2 * n + j];
            float cand = MathF.Tanh(nIh + r * nHh);
            next[j] = (1 - z) * cand + z * h[j];
        }
        return next;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // Accumulated in double: closer to the exact product than any float32 summation order numpy's BLAS might use.
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += (double)a[i] * b[i];
        return (float)sum;
    }

    private static G2pEnLetterToSound Load()
    {
        using Stream raw = EmbeddedTextResource.OpenGzip("g2p_en_lts.bin.gz");
        using MemoryStream ms = new();
        raw.CopyTo(ms);
        byte[] bytes = ms.ToArray();
        int pos = 0;
        string[] graphemes = ReadString(bytes, ref pos).Split('\t');
        string[] phonemes = ReadString(bytes, ref pos).Split('\t');
        Dictionary<string, Matrix> tensors = new(StringComparer.Ordinal);
        while (pos < bytes.Length)
        {
            string name = ReadString(bytes, ref pos);
            int ndim = ReadInt(bytes, ref pos);
            int rows = ReadInt(bytes, ref pos);
            int cols = ndim == 2 ? ReadInt(bytes, ref pos) : 1;
            if (ndim is < 1 or > 2) throw new InvalidDataException($"g2p_en tensor '{name}' has rank {ndim}.");
            float[] data = new float[rows * cols];
            for (int i = 0; i < data.Length; i++, pos += 4)
                data[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(pos, 4));
            tensors[name] = new Matrix(data, rows, cols);
        }
        return new G2pEnLetterToSound(graphemes, phonemes, tensors);
    }

    private static int ReadInt(byte[] bytes, ref int pos)
    {
        int v = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos, 4));
        pos += 4;
        return v;
    }

    private static string ReadString(byte[] bytes, ref int pos)
    {
        int len = ReadInt(bytes, ref pos);
        string s = Encoding.UTF8.GetString(bytes, pos, len);
        pos += len;
        return s;
    }

    private sealed class Matrix(float[] data, int rows, int cols)
    {
        public float[] Data { get; } = data;

        public int Rows { get; } = rows;

        public int Cols { get; } = cols;

        public ReadOnlySpan<float> Row(int r) => Data.AsSpan(r * Cols, Cols);
    }
}
