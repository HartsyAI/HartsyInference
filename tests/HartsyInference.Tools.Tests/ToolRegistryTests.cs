using System.Text.Json;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary><see cref="ToolRegistry"/>'s thread safety, snapshots, removal and the optional per-tool and default timeouts.</summary>
public sealed class ToolRegistryTests
{
    private static Task<string> Echo(string args, CancellationToken cancel) => Task.FromResult("ok");

    private static NativeToolCall Call(string name) => new() { Name = name, Arguments = "{}" };

    private static string? ErrorOf(string json) => JsonDocument.Parse(json).RootElement.GetProperty("error").GetString();

    [Fact]
    public void RemoveDropsToolKeepsOrderAndAllowsReAdd()
    {
        ToolRegistry registry = new();
        registry.Add("a", "", "{}", Echo).Add("b", "", "{}", Echo).Add("c", "", "{}", Echo);

        Assert.True(registry.Remove("b"));
        Assert.False(registry.Remove("b"));
        Assert.False(registry.TryGet("b", out _));
        Assert.Equal(["a", "c"], registry.Definitions.Select(d => d.Name));
        Assert.Equal(["a", "c"], registry.Names);
        Assert.Equal(2, registry.Count);

        registry.Add("b", "", "{}", Echo);
        Assert.Equal(["a", "c", "b"], registry.Names);
        Assert.Throws<ArgumentException>(() => registry.Add("a", "", "{}", Echo));
    }

    [Fact]
    public void DefinitionsSnapshotIsStableAcrossLaterChanges()
    {
        ToolRegistry registry = new();
        registry.Add("a", "", "{}", Echo).Add("b", "", "{}", Echo);
        IReadOnlyList<ToolDefinition> snapshot = registry.Definitions;
        IEnumerable<string> names = registry.Names;

        registry.Remove("a");
        registry.Add("c", "", "{}", Echo);

        Assert.Equal(["a", "b"], snapshot.Select(d => d.Name));
        Assert.Equal(["a", "b"], names);
        Assert.Equal(["b", "c"], registry.Definitions.Select(d => d.Name));
    }

    [Fact]
    public async Task ConcurrentAddRemoveAndEnumerationStayConsistent()
    {
        ToolRegistry registry = new();
        registry.Add("stable", "", "{}", Echo);
        const int writers = 4, perWriter = 200;
        List<Task> tasks = [];
        for (int w = 0; w < writers; w++)
        {
            int id = w;
            tasks.Add(Task.Run(() =>
            {
                for (int i = 0; i < perWriter; i++)
                {
                    string name = $"t{id}_{i}";
                    registry.Add(name, "", "{}", Echo);
                    if (i % 2 == 0) Assert.True(registry.Remove(name));
                }
            }));
        }
        for (int r = 0; r < 4; r++)
        {
            tasks.Add(Task.Run(async () =>
            {
                for (int i = 0; i < 500; i++)
                {
                    IReadOnlyList<ToolDefinition> defs = registry.Definitions;
                    int count = defs.Count;
                    Assert.Equal(count, defs.Count);
                    Assert.Equal(count, defs.Select(d => d.Name).Distinct().Count());
                    Assert.Equal("stable", defs[0].Name);
                    Assert.Equal("ok", await registry.InvokeAsync(Call("stable")));
                    registry.TryGet("t0_1", out _);
                }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.Equal(1 + writers * perWriter / 2, registry.Count);
        Assert.Equal(registry.Count, registry.Definitions.Count);
    }

    [Fact]
    public async Task PerToolTimeoutCancelsHandlerAndReturnsErrorJson()
    {
        ToolRegistry registry = new();
        bool cancelled = false;
        TaskCompletionSource observed = new(TaskCreationOptions.RunContinuationsAsynchronously);
        registry.Add("hang", "", "{}", async (_, c) =>
        {
            try { await Task.Delay(Timeout.Infinite, c); }
            catch (OperationCanceledException) { cancelled = true; observed.SetResult(); throw; }
            return "never";
        }, TimeSpan.FromMilliseconds(100));

        string result = await registry.InvokeAsync(Call("hang"));

        Assert.Contains("timed out", ErrorOf(result));
        await observed.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(cancelled);
    }

    [Fact]
    public async Task TimeoutBoundsHandlerThatIgnoresItsToken()
    {
        ToolRegistry registry = new() { DefaultTimeout = TimeSpan.FromMilliseconds(100) };
        using ManualResetEventSlim release = new();
        registry.Add("stuck", "", "{}", (_, _) => Task.Run(() => { release.Wait(); return "late"; }));
        registry.Add("blocking", "", "{}", (_, _) => { release.Wait(); return Task.FromResult("late"); });

        try
        {
            Assert.Contains("timed out", ErrorOf(await registry.InvokeAsync(Call("stuck")).WaitAsync(TimeSpan.FromSeconds(5))));
            Assert.Contains("timed out", ErrorOf(await registry.InvokeAsync(Call("blocking")).WaitAsync(TimeSpan.FromSeconds(5))));
        }
        finally
        {
            release.Set();
        }
    }

    [Fact]
    public async Task TurnCancellationStillThrowsWithTimeoutConfigured()
    {
        ToolRegistry registry = new() { DefaultTimeout = TimeSpan.FromSeconds(30) };
        registry.Add("hang", "", "{}", async (_, c) => { await Task.Delay(Timeout.Infinite, c); return "never"; });
        using CancellationTokenSource turn = new(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => registry.InvokeAsync(Call("hang"), turn.Token));
    }

    [Fact]
    public void NonPositiveTimeoutIsRejected()
    {
        ToolRegistry registry = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => registry.DefaultTimeout = TimeSpan.Zero);
        Assert.Throws<ArgumentOutOfRangeException>(() => registry.Add("x", "", "{}", Echo, TimeSpan.FromSeconds(-1)));
        registry.DefaultTimeout = TimeSpan.FromSeconds(1);
        registry.DefaultTimeout = null;
        Assert.Null(registry.DefaultTimeout);
    }
}
