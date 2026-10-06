using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random official audiocraft <c>StyleConditioner</c> (<c>tools/controlfoley/style_reference.py tiny</c>: tiny MERT
/// stand-in, sinusoidal pre-norm transformer, batch norm, RVQ with <c>eval_q=1</c>, downsample, projection, length mask) must
/// match the port stage by stage, and the julius resampler must keep a constant signal constant.</summary>
public sealed unsafe class ControlFoleyStyleParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoleyStyle");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void StyleTokens_MatchOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "style_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys)
        {
            weights[name] = loader.GetTensor(name);
        }

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "style_tiny.json")));
        JsonElement j = doc.RootElement;
        ControlFoleyStyleConfig config = new()
        {
            Mert = new HubertConfig
            {
                ConvDim = j.GetProperty("mert_conv").GetInt32(), Hidden = j.GetProperty("hidden").GetInt32(),
                NumLayers = j.GetProperty("mert_layers").GetInt32(), NumHeads = j.GetProperty("mert_heads").GetInt32(),
                FfnDim = j.GetProperty("mert_ffn").GetInt32(), PosConvKernel = j.GetProperty("pos_kernel").GetInt32(),
                PosConvGroups = j.GetProperty("pos_groups").GetInt32(),
            },
            Dim = j.GetProperty("dim").GetInt32(), Layers = j.GetProperty("layers").GetInt32(), Heads = j.GetProperty("heads").GetInt32(),
            Bins = j.GetProperty("bins").GetInt32(), Codebooks = j.GetProperty("codebooks").GetInt32(),
            EvalCodebooks = j.GetProperty("eval_q").GetInt32(), Downsample = j.GetProperty("ds").GetInt32(),
            OutputDim = j.GetProperty("out_dim").GetInt32(),
        };

        using IBackend backend = new CpuBackend();
        using ControlFoleyStyleEncoder encoder = new(config);
        encoder.LoadWeights(weights);
        for (int i = 0; i < j.GetProperty("cases").GetInt32(); i++)
        {
            float[] wave = Read(weights[$"in.wave{i}"]);
            float mert = ControlFoleyClapParityTests.MaxAbs(Read(weights[$"ref.mert{i}"]), encoder.EncodeMert(backend, wave, out int _));
            Assert.True(mert <= 1e-4f, $"case {i}: MERT max |d| = {mert}");
            float pre = ControlFoleyClapParityTests.MaxAbs(Read(weights[$"ref.pre{i}"]), encoder.EncodePreQuantizer(backend, wave, out int _));
            Assert.True(pre <= 2e-4f, $"case {i}: pre-quantiser max |d| = {pre}");
            float tokens = ControlFoleyClapParityTests.MaxAbs(Read(weights[$"ref.tokens{i}"]), encoder.Encode(backend, wave));
            Assert.True(tokens <= 2e-4f, $"case {i}: style tokens max |d| = {tokens}");
        }
    }

    [Fact]
    public void JuliusResampler_PreservesConstantsAndLength()
    {
        float[] ones = Enumerable.Repeat(0.5f, 6400).ToArray();
        float[] down = JuliusResampler.Resample(ones, 32_000, 24_000);
        Assert.Equal(4800, down.Length);
        Assert.All(down, v => Assert.InRange(v, 0.4999f, 0.5001f));
    }
}
