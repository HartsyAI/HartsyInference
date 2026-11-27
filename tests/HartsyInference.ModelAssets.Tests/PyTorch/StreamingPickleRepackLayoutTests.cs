using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

/// <summary>The two layouts a real checkpoint has that a flat state dict does not: tensors that are views of one storage (with an alias at the same offset), and a root that is a dict of state dicts.</summary>
public sealed class StreamingPickleRepackLayoutTests : IDisposable
{
    // torch.save({"a": base[:6].view(2, 3), "b": base[6:], "c": base[:6].view(2, 3)}) with base = arange(12, float32): one storage, "c" aliases "a".
    private const string Shared =
        "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgBAAAAYXEBY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQIoKFgHAAAAc3RvcmFn" +
        "ZXEDY3RvcmNoCkZsb2F0U3RvcmFnZQpxBFgBAAAAMHEFWAMAAABjcHVxBksMdHEHUUsASwJLA4ZxCEsDSwGGcQmJY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEKKVJxC3RxDFJxDVgBAAAAYnEOaAIoKGgDaARoBWgGSwx0" +
        "cQ9RSwZLBoVxEEsBhXERiWgKKVJxEnRxE1JxFFgBAAAAY3EVaAIoKGgDaARoBWgGSwx0cRZRSwBLAksDhnEXSwNLAYZxGIloCilScRl0cRpScRt1LlBLBwjdlLLmEQEAABEBAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAA" +
        "ABEAMABhcmNoaXZlL2J5dGVvcmRlckZCLABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APgBhcmNo" +
        "aXZlL2RhdGEvMEZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAAAAAgD8AAABAAABAQAAAgEAAAKBAAADAQAAA4EAAAABBAAAQQQAAIEEAADBBUEsHCHh9" +
        "Zj4wAAAAMAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADwATAGFyY2hpdmUvdmVyc2lvbkZCDwBaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4AMgBhcmNo" +
        "aXZlLy5kYXRhL3NlcmlhbGl6YXRpb25faWRGQi4AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjExNDY0NDI3MjM4NDkzNTUwNDE1MDAwMDE4MjA4MDA5Njc0Njk1MzFQSwcItBXQCSgA" +
        "AAAoAAAAUEsBAgAAAAAICAAAAAAAAN2UsuYRAQAAEQEAABAAAAAAAAAAAAAAAAAAAAAAAGFyY2hpdmUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAEQAAAAAAAAAAAAAAAABhAQAAYXJjaGl2ZS9ieXRl" +
        "b3JkZXJQSwECAAAAAAgIAAAAAAAAeH1mPjAAAAAwAAAADgAAAAAAAAAAAAAAAADWAQAAYXJjaGl2ZS9kYXRhLzBQSwECAAAAAAgIAAAAAAAA0Z5nVQIAAAACAAAADwAAAAAAAAAAAAAAAACAAgAAYXJjaGl2ZS92ZXJzaW9u" +
        "UEsBAgAAAAAICAAAAAAAALQV0AkoAAAAKAAAAB4AAAAAAAAAAAAAAAAA0gIAAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZFBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAUAAAAAAAAABQAAAAAAAABCAQAAAAAAAHgD" +
        "AAAAAAAAUEsGBwAAAAC6BAAAAAAAAAEAAABQSwUGAAAAAAUABQBCAQAAeAMAAAAA";

    // torch.save({"state_dict": {"model": {"w": tensor([1., 2.])}}, "callbacks": {"k": tensor([3, 4])}}): a dict of dicts, with an int64 tensor.
    private const string Nested =
        "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgKAAAAc3RhdGVfZGljdHEBfXECWAUAAABtb2RlbHEDfXEEWAEAAAB3cQVjdG9yY2guX3V0aWxz" +
        "Cl9yZWJ1aWxkX3RlbnNvcl92MgpxBigoWAcAAABzdG9yYWdlcQdjdG9yY2gKRmxvYXRTdG9yYWdlCnEIWAEAAAAwcQlYAwAAAGNwdXEKSwJ0cQtRSwBLAoVxDEsBhXENiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxDilS" +
        "cQ90cRBScRFzc1gJAAAAY2FsbGJhY2tzcRJ9cRNYAQAAAGtxFGgGKChoB2N0b3JjaApMb25nU3RvcmFnZQpxFVgBAAAAMXEWaApLAnRxF1FLAEsChXEYSwGFcRmJaA4pUnEadHEbUnEcc3UuUEsHCFif7bwoAQAAKAEAAFBL" +
        "AwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAEQAZAGFyY2hpdmUvYnl0ZW9yZGVyRkIVAFpaWlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APgBhcmNo" +
        "aXZlL2RhdGEvMEZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAACAPwAAAEBQSwcIdqU/LggAAAAIAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOADwA" +
        "YXJjaGl2ZS9kYXRhLzFGQjgAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloDAAAAAAAAAAQAAAAAAAAAUEsHCF3xOUEQAAAAEAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAA" +
        "AAAAAAAADwAzAGFyY2hpdmUvdmVyc2lvbkZCLwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgAyAGFy" +
        "Y2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZEZCLgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMDQ5NzA3Mjk5Mzc5OTEzNzQ2MjAwMDA5ODU3MjQ0NjA0OTEwNzA1NlBLBwgi18at" +
        "KAAAACgAAABQSwECAAAAAAgIAAAAAAAAWJ/tvCgBAAAoAQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAACFPeMZBgAAAAYAAAARAAAAAAAAAAAAAAAAAHgBAABhcmNoaXZlL2J5" +
        "dGVvcmRlclBLAQIAAAAACAgAAAAAAAB2pT8uCAAAAAgAAAAOAAAAAAAAAAAAAAAAANYBAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAABd8TlBEAAAABAAAAAOAAAAAAAAAAAAAAAAAFgCAABhcmNoaXZlL2RhdGEv" +
        "MVBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAAOACAABhcmNoaXZlL3ZlcnNpb25QSwECAAAAAAgIAAAAAAAAItfGrSgAAAAoAAAAHgAAAAAAAAAAAAAAAABSAwAAYXJjaGl2ZS8uZGF0YS9zZXJp" +
        "YWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAABgAAAAAAAAAGAAAAAAAAAH4BAAAAAAAA+AMAAAAAAABQSwYHAAAAAHYFAAAAAAAAAQAAAFBLBQYAAAAABgAGAH4BAAD4AwAAAAA=";

    private readonly string _folder = Path.Combine(Path.GetTempPath(), $"repack-layout-{Guid.NewGuid():N}");

    public StreamingPickleRepackLayoutTests() => Directory.CreateDirectory(_folder);

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string Place(string name, string base64)
    {
        string path = Path.Combine(_folder, name);
        File.WriteAllBytes(path, Convert.FromBase64String(base64));
        return path;
    }

    [Fact]
    public void ViewsOfOneStorageAndAnAliasAreWrittenAsSeparateTensorsAndVerifyBothWays()
    {
        string source = Place("shared.pt", Shared);
        string output = Path.Combine(_folder, "shared.safetensors");

        PickleRepackSummary summary = StreamingPickleRepack.Repack(source, output);

        Assert.Equal(3, summary.TensorCount);
        Assert.Equal(6 * 4 + 6 * 4 + 6 * 4, summary.TensorBytes);
        using SafeTensorsLoader loader = new();
        loader.Load(output);
        Assert.Equal(["a", "b", "c"], loader.Descriptors.Keys);
        Assert.Equal([0f, 1f, 2f, 3f, 4f, 5f], loader.GetTensor("a").AsSpan<float>().ToArray());
        Assert.Equal([6f, 7f, 8f, 9f, 10f, 11f], loader.GetTensor("b").AsSpan<float>().ToArray());
        Assert.Equal([0f, 1f, 2f, 3f, 4f, 5f], loader.GetTensor("c").AsSpan<float>().ToArray());
        Assert.True(StreamingPickleRepack.Verify(source, output).Equal);
        Assert.True(StreamingPickleRepack.Verify(source, output, wholeSource: true).Equal);
    }

    [Fact]
    public void ARereadOfAStorageEntryIsCountedButTheFirstOpeningOfEachEntryIsNot()
    {
        string source = Place("shared.pt", Shared);

        PickleRepackSummary summary = StreamingPickleRepack.Repack(source, Path.Combine(_folder, "shared.safetensors"));

        // One entry holds all three tensors; reading "c" (offset 0) after "b" goes back to the start of the entry.
        Assert.Equal(1, summary.StorageRestarts);
        PickleRepackSummary flat = StreamingPickleRepack.Repack(Place("flat.bin", PickleLazyReadTests.Checkpoint), Path.Combine(_folder, "flat.safetensors"));
        Assert.Equal(0, flat.StorageRestarts);
    }

    [Fact]
    public void ADictOfStateDictsIsReadRecursivelyWithDottedNamesIncludingIntegerTensors()
    {
        string source = Place("nested.ckpt", Nested);
        string output = Path.Combine(_folder, "nested.safetensors");

        PickleRepackSummary summary = StreamingPickleRepack.Repack(source, output, recursiveFlatten: true);

        Assert.Equal(2, summary.TensorCount);
        using SafeTensorsLoader loader = new();
        loader.Load(output);
        Assert.Equal(["state_dict.model.w", "callbacks.k"], loader.Descriptors.Keys);
        Assert.Equal(DType.I64, loader.Descriptors["callbacks.k"].DType);
        Assert.Equal([3L, 4L], loader.GetTensor("callbacks.k").AsSpan<long>().ToArray());
        Assert.True(StreamingPickleRepack.Verify(source, output, recursiveFlatten: true).Equal);
        Assert.True(StreamingPickleRepack.Verify(source, output, recursiveFlatten: true, wholeSource: true).Equal);
    }

    [Fact]
    public void AFileThatIsNotATorchCheckpointIsRejectedByTheWholeFileReaderWithoutLeavingAnOutput()
    {
        string notTorch = Path.Combine(_folder, "text.bin");
        File.WriteAllText(notTorch, "this is not a checkpoint");

        Assert.False(PytorchPickleLoader.IsZipCheckpoint(notTorch));
        Assert.Throws<InvalidOperationException>(() => StreamingPickleRepack.Repack(notTorch, Path.Combine(_folder, "never.safetensors")));
        Assert.False(File.Exists(Path.Combine(_folder, "never.safetensors")));
    }
}
