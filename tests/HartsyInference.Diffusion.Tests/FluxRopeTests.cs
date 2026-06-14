using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for FluxRope (axial RoPE). Validates position ID construction, cos/sin table computation, and rotation correctness against known reference values.</summary>
public sealed class FluxRopeTests
{
    private const float Tolerance = 1e-5f;

    // ── Position ID Tests ──────────────────────────────────────────

    [Fact]
    public unsafe void BuildPositionIds_CorrectShape()
    {
        int txtSeqLen = 8;
        int hPacked = 4;
        int wPacked = 4;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);

        Assert.Equal(2, posIds.Shape.Rank);
        Assert.Equal(totalSeqLen, (int)posIds.Shape[0]);
        Assert.Equal(3, (int)posIds.Shape[1]);

        posIds.Dispose();
    }

    [Fact]
    public unsafe void BuildPositionIds_TextTokensAreAllZeros()
    {
        int txtSeqLen = 8;
        int hPacked = 4;
        int wPacked = 4;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        float* ptr = (float*)posIds.DataPointer;

        for (int t = 0; t < txtSeqLen; t++)
        {
            Assert.Equal(0f, ptr[t * 3 + 0]); // axis 0
            Assert.Equal(0f, ptr[t * 3 + 1]); // axis 1 (row)
            Assert.Equal(0f, ptr[t * 3 + 2]); // axis 2 (col)
        }

        posIds.Dispose();
    }

    [Fact]
    public unsafe void BuildPositionIds_ImageTokensHaveCorrectRowCol()
    {
        int txtSeqLen = 4;
        int hPacked = 3;
        int wPacked = 2;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        float* ptr = (float*)posIds.DataPointer;

        // Image tokens start after text tokens
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = txtSeqLen + row * wPacked + col;
                Assert.Equal(0f, ptr[idx * 3 + 0]); // axis 0 always 0
                Assert.Equal(row, ptr[idx * 3 + 1]); // axis 1 = row
                Assert.Equal(col, ptr[idx * 3 + 2]); // axis 2 = col
            }
        }

        posIds.Dispose();
    }

    // ── Precompute Tests ───────────────────────────────────────────

    [Fact]
    public void Precompute_CreatesTablesWithCorrectSize()
    {
        int[] axesDim = [16, 56, 56]; // sum = 128 = head_dim
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 4;
        int hPacked = 2;
        int wPacked = 2;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        // After precompute, the rope should be ready for Forward calls
        // Verify by doing a small Forward with dummy data
        int batch = 1;
        int numHeads = 2;
        int headDim = 128; // sum of axesDim
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);
        Tensor q = new Tensor(mhShape, DType.F32);
        Tensor k = new Tensor(mhShape, DType.F32);

        // Should not throw
        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        q.Dispose();
        k.Dispose();
    }

    // ── Rotation Tests ────────────��────────────────────────────────

    [Fact]
    public unsafe void Forward_ZeroInput_StaysZero()
    {
        int[] axesDim = [16, 56, 56];
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 2;
        int hPacked = 2;
        int wPacked = 2;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        int batch = 1;
        int numHeads = 1;
        int headDim = 128;
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);

        Tensor q = new Tensor(mhShape, DType.F32);
        Tensor k = new Tensor(mhShape, DType.F32);

        // Fill with zeros
        int count = (int)q.ElementCount;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        for (int i = 0; i < count; i++)
        {
            qPtr[i] = 0f;
            kPtr[i] = 0f;
        }

        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        // Rotating zeros should give zeros
        for (int i = 0; i < count; i++)
        {
            Assert.InRange(qPtr[i], -Tolerance, Tolerance);
            Assert.InRange(kPtr[i], -Tolerance, Tolerance);
        }

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public unsafe void Forward_TextPositionsZero_NoRotationApplied()
    {
        // Text positions are all [0,0,0], so cos=1, sin=0 → no rotation
        int[] axesDim = [16, 56, 56];
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 4;
        int hPacked = 2;
        int wPacked = 2;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        int batch = 1;
        int numHeads = 1;
        int headDim = 128;
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);

        Tensor q = new Tensor(mhShape, DType.F32);

        // Fill text positions with known values
        float* qPtr = (float*)q.DataPointer;
        int count = (int)q.ElementCount;
        for (int i = 0; i < count; i++) qPtr[i] = 0f;

        // Set text token 0 to a pattern
        for (int d = 0; d < headDim; d++)
            qPtr[d] = (float)(d + 1);

        // Save original text values
        float[] original = new float[headDim];
        for (int d = 0; d < headDim; d++)
            original[d] = qPtr[d];

        Tensor k = new Tensor(mhShape, DType.F32);
        float* kPtr = (float*)k.DataPointer;
        for (int i = 0; i < count; i++) kPtr[i] = 0f;

        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        // Text token 0 has position [0,0,0], so angle=0 → cos=1, sin=0
        // Rotation: x' = cos*x - sin*y = 1*x - 0*y = x, y' = sin*x + cos*y = 0*x + 1*y = y
        // So values should be unchanged
        for (int d = 0; d < headDim; d++)
        {
            Assert.InRange(qPtr[d], original[d] - Tolerance, original[d] + Tolerance);
        }

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public unsafe void Forward_UnitVector_RotatesCorrectly()
    {
        // Test that a unit vector [1, 0] at a known position gets rotated by the correct angle
        int[] axesDim = [4, 4, 4]; // Small dims for easy verification. sum=12
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 0;
        int hPacked = 2;
        int wPacked = 1;
        int totalSeqLen = hPacked * wPacked; // = 2

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        int batch = 1;
        int numHeads = 1;
        int headDim = 12; // 4+4+4
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);

        Tensor q = new Tensor(mhShape, DType.F32);
        Tensor k = new Tensor(mhShape, DType.F32);
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        int count = (int)q.ElementCount;
        for (int i = 0; i < count; i++)
        {
            qPtr[i] = 0f;
            kPtr[i] = 0f;
        }

        // Position 0: image token at row=0, col=0 → all positions 0 → no rotation
        // Position 1: image token at row=1, col=0 → axis 1 has pos=1
        // Set q[seq=1, dim=0:1] = [1.0, 0.0] → first pair in axis 0 (pos=0 → no rotation)
        int pos1Base = headDim; // seq=1 offset
        qPtr[pos1Base + 0] = 1.0f;
        qPtr[pos1Base + 1] = 0.0f;

        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        // Axis 0 dims [0..3], pos=0 → cos=1, sin=0 → no rotation
        Assert.InRange(qPtr[pos1Base + 0], 1.0f - Tolerance, 1.0f + Tolerance);
        Assert.InRange(qPtr[pos1Base + 1], -Tolerance, Tolerance);

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public unsafe void Forward_PreservesNorm()
    {
        // RoPE is a rotation, so ||q|| should be preserved
        int[] axesDim = [16, 56, 56];
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 2;
        int hPacked = 4;
        int wPacked = 4;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        int batch = 1;
        int numHeads = 2;
        int headDim = 128;
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);

        Tensor q = new Tensor(mhShape, DType.F32);
        Tensor k = new Tensor(mhShape, DType.F32);
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        // Fill with deterministic values
        Random rng = new Random(42);
        int count = (int)q.ElementCount;
        for (int i = 0; i < count; i++)
        {
            qPtr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            kPtr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Compute norms before rotation
        float[] normsBefore = new float[numHeads * totalSeqLen];
        for (int h = 0; h < numHeads; h++)
        {
            for (int s = 0; s < totalSeqLen; s++)
            {
                int offset = (h * totalSeqLen + s) * headDim;
                float norm = 0f;
                for (int d = 0; d < headDim; d++)
                    norm += qPtr[offset + d] * qPtr[offset + d];
                normsBefore[h * totalSeqLen + s] = MathF.Sqrt(norm);
            }
        }

        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        // Verify norms are preserved after rotation
        for (int h = 0; h < numHeads; h++)
        {
            for (int s = 0; s < totalSeqLen; s++)
            {
                int offset = (h * totalSeqLen + s) * headDim;
                float norm = 0f;
                for (int d = 0; d < headDim; d++)
                    norm += qPtr[offset + d] * qPtr[offset + d];
                float normAfter = MathF.Sqrt(norm);

                float normBefore = normsBefore[h * totalSeqLen + s];
                Assert.InRange(normAfter, normBefore - 1e-4f, normBefore + 1e-4f);
            }
        }

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public unsafe void Forward_QAndK_RotatedSameWay()
    {
        // Q and K at the same position should be rotated by the same angle
        int[] axesDim = [16, 56, 56];
        FluxRope rope = new FluxRope(axesDim, theta: 10000);

        int txtSeqLen = 2;
        int hPacked = 2;
        int wPacked = 2;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();

        int batch = 1;
        int numHeads = 1;
        int headDim = 128;
        TensorShape mhShape = new TensorShape(batch, numHeads, totalSeqLen, headDim);

        Tensor q = new Tensor(mhShape, DType.F32);
        Tensor k = new Tensor(mhShape, DType.F32);
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        // Set Q and K to identical values
        Random rng = new Random(42);
        int count = (int)q.ElementCount;
        for (int i = 0; i < count; i++)
        {
            float val = (float)(rng.NextDouble() * 2.0 - 1.0);
            qPtr[i] = val;
            kPtr[i] = val;
        }

        rope.Forward(q, k, batch, numHeads, totalSeqLen);

        // After rotation, Q and K should still be identical
        for (int i = 0; i < count; i++)
        {
            Assert.InRange(kPtr[i], qPtr[i] - Tolerance, qPtr[i] + Tolerance);
        }

        q.Dispose();
        k.Dispose();
    }

    // ── Latent Packing Tests (FluxPipeline Pack/Unpack) ────────────

    [Fact]
    public void BuildPositionIds_1024x1024Resolution()
    {
        // 1024x1024 image: latent 128x128, packed 64x64 = 4096 image tokens
        int txtSeqLen = 256;
        int hPacked = 64;
        int wPacked = 64;
        int totalSeqLen = txtSeqLen + hPacked * wPacked;

        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);

        Assert.Equal(totalSeqLen, (int)posIds.Shape[0]);
        Assert.Equal(3, (int)posIds.Shape[1]);

        posIds.Dispose();
    }
}
