using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The head-major route of <see cref="AttentionLayer.Attend"/> ends with a permute of the attention output back to the
/// token-major layout. When that permute throws, the output of the attention and the tensor the permute was to fill must both be
/// released and the exception must reach the caller. The throw comes from a forwarding proxy that refuses the fourth permute (the
/// three before it lay out the operands), so no fault-injection hook is involved. A tensor is disposed when reading its data throws
/// <see cref="ObjectDisposedException"/>.</summary>
public sealed class AttentionLayerRunHeadMajorTests
{
    private const int Heads = 3, HeadDim = 5, Tokens = 7;

    private static Tensor TokenMajor()
    {
        Tensor tensor = new(new TensorShape(1, Tokens, Heads, HeadDim), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (i % 11 - 5) / 4f;
        }

        return tensor;
    }

    private static bool IsDisposed(Tensor tensor)
    {
        try
        {
            _ = tensor.AsReadOnlySpan<float>();
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    /// <summary>Forwards every call to a real CPU backend; the permute with the number <see cref="RefusedPermute"/> records its two
    /// tensors and throws.</summary>
    public class PermuteRefusingBackend : DispatchProxy
    {
        public IBackend Target { get; set; } = null!;

        public int RefusedPermute { get; set; }

        public int Permutes { get; private set; }

        public Tensor? RefusedOutput { get; private set; }

        public Tensor? RefusedInput { get; private set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod!.Name == nameof(IBackend.Permute0213) && ++Permutes == RefusedPermute)
            {
                RefusedOutput = (Tensor)args![0]!;
                RefusedInput = (Tensor)args![1]!;
                throw new InvalidOperationException("the permute was refused");
            }

            try
            {
                return targetMethod.Invoke(Target, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }
    }

    private static (IBackend Proxy, PermuteRefusingBackend Control, CpuBackend Cpu) Proxied(int refusedPermute)
    {
        CpuBackend cpu = new();
        IBackend proxy = DispatchProxy.Create<IBackend, PermuteRefusingBackend>();
        PermuteRefusingBackend control = (PermuteRefusingBackend)proxy;
        control.Target = cpu;
        control.RefusedPermute = refusedPermute;
        return (proxy, control, cpu);
    }

    [Fact]
    public void WhenTheClosingPermuteThrowsTheAttentionOutputAndTheNewTensorAreDisposed()
    {
        (IBackend proxy, PermuteRefusingBackend control, CpuBackend cpu) = Proxied(refusedPermute: 4);
        using (cpu)
        {
            AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);

            InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
                () => layer.Attend(proxy, TokenMajor(), TokenMajor(), TokenMajor(), new TensorShape(1, Tokens, Heads * HeadDim), default));

            Assert.Equal("the permute was refused", ex.Message);
            Assert.Equal(4, control.Permutes);
            Tensor attention = control.RefusedInput ?? throw new Xunit.Sdk.XunitException("the closing permute was never reached");
            Tensor created = control.RefusedOutput ?? throw new Xunit.Sdk.XunitException("the closing permute was never reached");
            Assert.True(IsDisposed(attention), "the attention output was not disposed after the closing permute threw");
            Assert.True(IsDisposed(created), "the tensor the closing permute was to fill was not disposed");
        }
    }

    [Fact]
    public void TheClosingPermuteIsTheFourthOnTheHeadMajorRoute()
    {
        (IBackend proxy, PermuteRefusingBackend control, CpuBackend cpu) = Proxied(refusedPermute: 0);
        using (cpu)
        {
            AttentionLayer layer = AttentionLayer.Shared(Heads, HeadDim);
            using Tensor result = layer.Attend(proxy, TokenMajor(), TokenMajor(), TokenMajor(), new TensorShape(1, Tokens, Heads * HeadDim), default);

            Assert.Equal(4, control.Permutes);
            Assert.Equal(new TensorShape(1, Tokens, Heads * HeadDim), result.Shape);
        }
    }
}
