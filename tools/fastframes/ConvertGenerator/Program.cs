// Converts the Causal Forcing++ generator (a torch pickle) to safetensors through the engine's safe-subset pickle reader.
// The pickle is never executed: PytorchEnvelopeReader reads only the tensors under the generator_ema envelope, as F32 copies.
// The output keeps every key under generator_ema with that envelope name removed, so the dumper's loader reads it the same way
// it reads the pickle (its _strip removes the model._fsdp_wrapped_module. prefix).
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: ConvertGenerator <framewise-2step.pt> <out.safetensors>");
    return 2;
}

string input = Path.GetFullPath(args[0]);
string output = Path.GetFullPath(args[1]);
if (Path.GetFullPath(output) == input)
{
    Console.Error.WriteLine("refusing to write over the input; the original is kept.");
    return 2;
}
Directory.CreateDirectory(Path.GetDirectoryName(output)!);

(Dictionary<string, Tensor> weights, IDisposable owner) = PytorchEnvelopeReader.Read(input, "generator_ema", string.Empty, DType.F32);
using (owner)
{
    string digest = SafeTensorsMerger.Write(output, weights.ToList());
    Console.WriteLine($"tensors: {weights.Count}");
    Console.WriteLine($"output: {output}");
    Console.WriteLine($"payload sha256: {digest}");
}
return 0;
