using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Decision.Clef;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Runs the released Clef-Flash weights on the README's examples. Needs <c>HARTSY_CLEF_DIR</c> (the downloaded release);
/// without it the test does nothing. <c>HARTSY_CLEF_REQUEST</c> may name a JSON request file to use instead.</summary>
public sealed class ClefRealWeightTests(ITestOutputHelper output)
{
    private const string Invoice = """
        {"model":"clef-flash","state":{"invoice":{"vendor":"Acme","total":1250.0,"currency":"USD","status":"overdue"}},
         "questions":{
           "status":{"type":"choice","instructions":"What is the invoice status?","criteria":{"paid":"Invoice is paid.","overdue":"Invoice is past due.","draft":"Not sent."}},
           "large":{"type":"noul","instructions":"Is the total above 1000 USD?"}}}
        """;

    private const string Support = """
        {"model":"clef-flash","state":"Our checkout started returning errors and orders are blocked.",
         "questions":{
           "department":{"type":"choice","instructions":"Which team should handle the message?","criteria":{"billing":"Payments or invoices","technical":"Bugs or outages"}},
           "urgency":{"type":"score","criteria":["Can wait","This week","Today"]},
           "outage":{"type":"noul","instructions":"Is a service down?"}}}
        """;

    [Fact]
    public void Decisions_MatchTheReadmeExamples()
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_CLEF_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }
        List<SafeTensorsLoader> loaders = [];
        Dictionary<string, Tensor> backbone = new(StringComparer.Ordinal);
        foreach (string shard in Directory.GetFiles(dir, "model-*.safetensors").Order())
        {
            SafeTensorsLoader loader = new();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (string name in loader.Descriptors.Keys)
            {
                if (name.StartsWith("model.language_model.", StringComparison.Ordinal) || name == "lm_head.weight")
                {
                    backbone[name] = loader.GetTensor(name);
                }
            }
        }
        SafeTensorsLoader headLoader = new();
        headLoader.Load(Path.Combine(dir, "joint_head.safetensors"));
        loaders.Add(headLoader);
        Dictionary<string, Tensor> head = headLoader.Descriptors.Keys.ToDictionary(k => k, headLoader.GetTensor);
        using JsonDocument headConfig = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "joint_head_config.json")));
        JsonElement hc = headConfig.RootElement;
        ClefJointHeadConfig headCfg = new()
        {
            HiddenSize = hc.GetProperty("hidden_size").GetInt32(), Width = hc.GetProperty("width").GetInt32(),
            RoutingLayers = hc.GetProperty("routing_layers").GetInt32(), Layers = hc.GetProperty("layers").GetInt32(),
            Heads = hc.GetProperty("heads").GetInt32(), Feedforward = hc.GetProperty("feedforward").GetInt32(),
        };
        using FileStream tokenizerStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(File.ReadAllText(Path.Combine(dir, "config.json")));

        using CpuBackend backend = new();
        using ClefDecisionPipeline pipeline = ClefDecisionPipeline.Create(backbone, head, cfg, headCfg, tokenizer, maxLength: 2_048);
        try
        {
            string? custom = Environment.GetEnvironmentVariable("HARTSY_CLEF_REQUEST");
            if (custom is not null)
            {
                output.WriteLine(pipeline.Decide(backend, File.ReadAllText(custom)));
                return;
            }
            using JsonDocument invoice = JsonDocument.Parse(pipeline.Decide(backend, Invoice));
            output.WriteLine(invoice.RootElement.GetRawText());
            JsonElement invoiceAnswers = invoice.RootElement.GetProperty("answers");
            Assert.Equal("overdue", invoiceAnswers.GetProperty("status").GetProperty("choice").GetString());
            Assert.True(invoiceAnswers.GetProperty("large").GetProperty("noul").GetDouble() > 0.5);

            using JsonDocument support = JsonDocument.Parse(pipeline.Decide(backend, Support));
            output.WriteLine(support.RootElement.GetRawText());
            JsonElement supportAnswers = support.RootElement.GetProperty("answers");
            Assert.Equal("technical", supportAnswers.GetProperty("department").GetProperty("choice").GetString());
            Assert.True(supportAnswers.GetProperty("outage").GetProperty("noul").GetDouble() > 0.5);
            Assert.True(supportAnswers.GetProperty("urgency").GetProperty("score").GetDouble() > 1.0);
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders)
            {
                l.Dispose();
            }
        }
    }
}
