using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights forwards for the Fish-Speech / OpenAudio S1 codecs and tokenizer: the firefly
/// grouped-residual FSQ quantizer (codes → finite 1024-dim decoder input → SiLU HiFi-GAN), the modded-DAC RVQ
/// codec (codes → finite 44.1 kHz audio), and the BPE tokenizer round-tripping special tokens off an in-memory
/// vocab. CpuBackend + deterministic random tensors (mirrors <see cref="FishSpeechTests"/>).</summary>
public sealed unsafe class FishSpeechCodecTests
{
    private static uint _rng = 0xDAC0u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor Alpha(int n) { Tensor t = new(new TensorShape(1, n, 1), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 0.5f; return t; }

    internal static int FsqVocab(int[] levels) => Fsq.VocabSize(levels);

    [Fact]
    public void FireflyQuantizer_SyntheticForward_CodesToFiniteDecoderInput()
    {
        int[] levels = [2, 2];
        int nq = 3;
        using CpuBackend backend = new();
        FireflyQuantizer q = new(levels, nq, [2]);
        q.LoadWeights(QuantizerWeights(levels, qUp: [2]), "quantizer");
        int t = 4;
        int vocab = FsqVocab(levels);
        int[,] codes = new int[nq, t];
        for (int i = 0; i < nq; i++) for (int j = 0; j < t; j++) codes[i, j] = (i * 2 + j) % vocab;
        using Tensor decoderInput = q.DequantToDecoderInput(backend, codes, t);
        Assert.Equal(1, (int)decoderInput.Shape[0]);
        Assert.Equal(1_024, (int)decoderInput.Shape[1]);
        Assert.Equal(t * q.UpsampleFactor, (int)decoderInput.Shape[2]);
        float* p = (float*)decoderInput.DataPointer;
        for (long n = 0; n < decoderInput.ElementCount; n++) Assert.True(float.IsFinite(p[n]));
    }

    [Fact]
    public void FishDacCodec_SyntheticForward_CodesToFiniteAudio()
    {
        // Tiny DAC: latent 8, decoderDim 16, rates [2,2], dilations [1,3]. 1 semantic + 2 residual books.
        int numResidual = 2, codebookDim = 4, latentDim = 8, decoderDim = 16;
        int[] rates = [2, 2];
        int[] dilations = [1, 3];
        using CpuBackend backend = new();
        FishDacCodec codec = new(numResidual, codebookDim, latentDim, decoderDim, rates, dilations, 44_100);
        codec.LoadWeights(DacWeights(numResidual, codebookDim, latentDim, decoderDim, rates, dilations));
        int t = 4;
        int books = 1 + numResidual;
        int[,] codes = new int[books, t];
        for (int j = 0; j < t; j++)
        {
            codes[0, j] = (j * 7) % 4_096;
            for (int i = 1; i < books; i++) codes[i, j] = (i + j) % 1_024;
        }
        float[] audio = codec.Decode(backend, codes, t);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    [Fact]
    public void Tokenizer_RoundTripsSpecialsAndText()
    {
        Dictionary<string, int> vocab = new()
        {
            ["<|im_start|>"] = 0,
            ["<|im_end|>"] = 1,
            ["<|audio_start|>"] = 2,
            ["<|audio_end|>"] = 3,
            ["<|semantic:0|>"] = 4,
            ["<|semantic:1|>"] = 5,
            ["<|semantic:2|>"] = 6,
        };
        // Single-byte tokens for every byte that "hi" uses, plus a couple merges to exercise BPE.
        int next = 100;
        foreach (char c in "hiabc ")
            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(c.ToString()))
                vocab.TryAdd(ByteSymbol(b), next++);

        FishSpeechTokenizer tok = new();
        tok.LoadFromVocab(vocab, merges: null,
            specials: ["<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>", "<|semantic:0|>"]);

        Assert.Equal(0, tok.ImStartId);
        Assert.Equal(1, tok.ImEndId);
        Assert.Equal(4, tok.SemanticBeginId);
        Assert.Equal(6, tok.SemanticId(2));
        Assert.Equal(2, tok.SpecialId("<|audio_start|>"));

        int[] ids = tok.Encode("<|im_start|>hi<|im_end|>");
        Assert.Equal(0, ids[0]);
        Assert.Equal(1, ids[^1]);
        Assert.Contains(ids, id => id >= 100);

        // Decode drops specials and reconstructs the byte payload.
        string decoded = tok.Decode(ids);
        Assert.Equal("hi", decoded);
    }

    [Fact]
    public void Tokenizer_LoadsTiktokenFile_MergesByRankAndRoundTrips()
    {
        // A minimal tiktoken file: each line is "<base64-bytes> <rank>". Single bytes for "hi" plus the merged
        // token "hi" at a higher rank, so BPE must prefer the longest merge available.
        static string B64(string s) => Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(s));
        string path = Path.Combine(Path.GetTempPath(), $"fish_{Guid.NewGuid():N}.tiktoken");
        string[] lines =
        [
            $"{B64("h")} 0",
            $"{B64("i")} 1",
            $"{B64("a")} 2",
            $"{B64("hi")} 3",   // merged token: rank 3 > singles, but BPE merges any available pair
        ];
        File.WriteAllLines(path, lines);
        try
        {
            FishSpeechTokenizer tok = new();
            // Specials are not in the tiktoken file; supply them (assigned ids after the base vocab).
            tok.LoadTiktoken(path, ["<|im_start|>", "<|im_end|>", "<|semantic:0|>", "<|semantic:1|>"]);

            Assert.Equal(4, tok.SpecialId("<|im_start|>"));   // base ranks 0..3 → specials start at 4
            Assert.Equal(5, tok.ImEndId);
            Assert.Equal(6, tok.SemanticBeginId);
            Assert.Equal(7, tok.SemanticId(1));

            // "hi" merges to the single id-3 token; specials are peeled off intact.
            int[] ids = tok.Encode("<|im_start|>hi<|im_end|>");
            Assert.Equal(new[] { 4, 3, 5 }, ids);

            // Decode drops specials and reconstructs the byte payload.
            Assert.Equal("hi", tok.Decode(ids));

            // Plain Load() must auto-dispatch to the tiktoken branch by extension.
            FishSpeechTokenizer viaLoad = new();
            viaLoad.Load(path);
            Assert.Equal(new[] { 3 }, viaLoad.Encode("hi"));
        }
        finally { File.Delete(path); }
    }

    private static string ByteSymbol(byte b)
    {
        // Mirror FishSpeechTokenizer's GPT-2 byte→unicode table for single-byte tokens.
        int[] printable = BuildPrintable();
        return printable[b] >= 0 ? ((char)printable[b]).ToString() : ((char)b).ToString();
    }

    private static int[] BuildPrintable()
    {
        List<int> bs = [];
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);
        List<int> cs = [.. bs];
        int n = 0;
        int[] map = new int[256];
        for (int i = 0; i < 256; i++) map[i] = -1;
        List<int> bsAll = [.. bs];
        for (int b = 0; b < 256; b++)
            if (!bs.Contains(b)) { bsAll.Add(b); cs.Add(256 + n); n++; }
        for (int i = 0; i < bsAll.Count; i++) map[bsAll[i]] = cs[i];
        return map;
    }

    internal static Dictionary<string, Tensor> QuantizerWeights(int[] levels, int[] qUp)
    {
        // codebookDim = levels.Length; no project_out (identity FSQ). fc_post: codebookDim → 8 channels (1×1).
        int codebookDim = levels.Length;
        Dictionary<string, Tensor> w = new()
        {
            ["quantizer.fc_post.weight"] = F3(8, codebookDim, 1),
            ["quantizer.fc_post.bias"] = F1(8),
        };
        int inCh = 8;
        for (int i = 0; i < qUp.Length; i++)
        {
            int outCh = inCh / 2;
            // ConvTranspose1d weight [C_in, C_out, K]; K = 2*stride.
            w[$"quantizer.upsample.{i}.weight"] = F3(inCh, outCh, 2 * qUp[i]);
            w[$"quantizer.upsample.{i}.bias"] = F1(outCh);
            inCh = outCh;
        }
        return w;
    }

    internal static Dictionary<string, Tensor> FireflyWeights(int[] levels, int[] qUp, int genInit, int[] genRates, int[] genKernels)
    {
        Dictionary<string, Tensor> w = new(QuantizerWeights(levels, qUp))
        {
            // Gen conv_pre consumes the 1024-channel decoder input.
            ["head.conv_pre.weight"] = F3(genInit, 1_024, 7),
            ["head.conv_pre.bias"] = F1(genInit),
            ["head.conv_post.weight"] = F3(1, genInit >> genRates.Length, 7),
            ["head.conv_post.bias"] = F1(1),
        };
        for (int i = 0; i < genRates.Length; i++)
        {
            int inCh = genInit >> i, outCh = inCh >> 1;
            w[$"head.ups.{i}.weight"] = F3(inCh, outCh, genKernels[i]);
            w[$"head.ups.{i}.bias"] = F1(outCh);
            w[$"head.resblocks.{i}.convs1.0.weight"] = F3(outCh, outCh, 3);
            w[$"head.resblocks.{i}.convs1.0.bias"] = F1(outCh);
            w[$"head.resblocks.{i}.convs2.0.weight"] = F3(outCh, outCh, 3);
            w[$"head.resblocks.{i}.convs2.0.bias"] = F1(outCh);
        }
        return w;
    }

    private static Dictionary<string, Tensor> DacWeights(int numResidual, int codebookDim, int latentDim, int decoderDim,
        int[] rates, int[] dilations)
    {
        Dictionary<string, Tensor> w = new();
        int books = 1 + numResidual;
        for (int i = 0; i < books; i++)
        {
            int size = i == 0 ? 4_096 : 1_024;
            w[$"quantizer.quantizers.{i}.codebook.weight"] = F2(size, codebookDim);
            w[$"quantizer.quantizers.{i}.out_proj.weight"] = F2(latentDim, codebookDim);
            w[$"quantizer.quantizers.{i}.out_proj.bias"] = F1(latentDim);
        }
        w["decoder.model.0.weight"] = F3(decoderDim, latentDim, 7);
        w["decoder.model.0.bias"] = F1(decoderDim);
        int dim = decoderDim;
        for (int i = 0; i < rates.Length; i++)
        {
            int outDim = dim / 2;
            w[$"decoder.model.{i + 1}.block.0.alpha"] = Alpha(dim);
            w[$"decoder.model.{i + 1}.block.1.weight"] = F3(dim, outDim, 2 * rates[i]);
            w[$"decoder.model.{i + 1}.block.1.bias"] = F1(outDim);
            for (int j = 0; j < dilations.Length; j++)
            {
                string p = $"decoder.model.{i + 1}.block.{j + 2}";
                // SnakeResBlock keys: weight-normed convs1/convs2 + activations1/activations2.alpha.
                AddWeightNormConv(w, $"{p}.convs1.0", outDim, outDim, 7);
                AddWeightNormConv(w, $"{p}.convs2.0", outDim, outDim, 1);
                w[$"{p}.activations1.0.alpha"] = Alpha(outDim);
                w[$"{p}.activations2.0.alpha"] = Alpha(outDim);
            }
            dim = outDim;
        }
        w[$"decoder.model.{rates.Length + 1}.alpha"] = Alpha(dim);
        w[$"decoder.model.{rates.Length + 2}.weight"] = F3(1, dim, 7);
        w[$"decoder.model.{rates.Length + 2}.bias"] = F1(1);
        return w;
    }

    private static void AddWeightNormConv(Dictionary<string, Tensor> w, string prefix, int outCh, int inCh, int k)
    {
        // weight_g [outCh,1,1], weight_v [outCh,inCh,k]; WeightNorm.Compose fuses to [outCh,inCh,k].
        w[$"{prefix}.weight_g"] = F3(outCh, 1, 1);
        w[$"{prefix}.weight_v"] = F3(outCh, inCh, k);
        w[$"{prefix}.bias"] = F1(outCh);
    }
}
