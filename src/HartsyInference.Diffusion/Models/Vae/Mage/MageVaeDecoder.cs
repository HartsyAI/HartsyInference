using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>Decoder for Microsoft Mage-Flow's bespoke <b>MageVAE</b> — a 128-channel, /16 <b>one-step diffusion
/// codec</b> (NOT an AutoencoderKL). Decodes a latent <c>[B, 128, H/16, W/16]</c> to RGB <c>[B, 3, H, W]</c> in a
/// single deterministic pass at fixed timestep <c>t = 0</c> with zero pixel-noise. Ported from
/// <c>github.com/microsoft/Mage · mage_flow/models/modules/mage_vae.py</c> (<c>MageVAE.decode</c> + <c>_DConvDenoiser</c>).
///
/// <para><b>Dataflow</b> (h = H/16, w = W/16):
/// <list type="number">
/// <item><c>cond = y_embedder.decoder(z)</c> — a CoD decoder (conv_in 128→384, [ResNet, Attn, ResNet, Attn, ResNet],
/// GroupNorm→SiLU→conv_out) staying at LATENT resolution → <c>[B, 384, h, w]</c>.</item>
/// <item><c>s = s_embedder([0 ; cond])</c> — proj1(zero-noise)=0, so s = proj2(cat 512→384) → <c>[B, 384, h, w]</c>.</item>
/// <item><c>s = 21 × DiCoBlock(s, c)</c> where <c>c = t_embedder(0)</c> (constant) — trunk at latent res.</item>
/// <item>NeRF per-patch decode: each of the h·w tokens emits a 16×16 RGB tile via DCT coordinate features +
/// SimpleMLPAdaLN, then <c>fold</c> tiles them into the full <c>[B, 3, H, W]</c> image.</item>
/// </list></para>
///
/// <para><b>Why it upsamples with no transposed conv:</b> the trunk runs entirely at latent res; the 16× is a
/// NeRF-style per-patch decode (h·w spatial tokens × 256 sub-pixels each) folded into the image.</para>
///
/// <para><b>Parity traps (from the reference — verify on GPU):</b> (1) NeRF freqs = <c>linspace(0,8,8)</c> (endpoint
/// inclusive, step 8/7 — NOT arange); (2) DiCoBlock norm1/norm2 are affine=false (channel-dim LayerNorm2d, no
/// params); (3) DiCoBlock conv2 is depthwise (groups=384); (4) final cast is <c>.byte()</c> truncation, not round;
/// (5) the 8960→[35,256]→[256,35] reshape/permute ordering; (6) fold kernel=stride=16 non-overlapping ordering.
/// The CoD <c>AttnBlock</c> is patched (patch_size 32) in the reference — this port uses GLOBAL attention
/// (<see cref="VaeAttention"/>), exact for latent grids ≤32 (images ≤512px) and an approximation above; flagged for
/// GPU parity.</para></summary>
public sealed unsafe class MageVaeDecoder : IDisposable
{
    public const int LatentChannels = 128;
    public const int Hidden = 384;        // trunk / cond width
    public const int HiddenX = 32;        // per-sub-pixel NeRF feature width
    public const int Patch = 16;          // VAE patch (= downsample factor)
    public const int SubPixels = Patch * Patch;   // 256 sub-pixels per token
    public const int MaxFreqs = 8;        // NeRF coordinate frequencies → MaxFreqs² = 64 coords
    public const int NumDiCoBlocks = 21;
    public const int DecResBlocks = 3;    // SimpleMLPAdaLN residual blocks (num_blocks 24 − 21)
    private const int GroupNormGroups = 32;
    private const float GroupNormEps = 1e-6f;

    // ── CoD decoder (y_embedder.decoder): latent → 384-ch conditioning at latent res ──
    private Tensor? _condConvInW, _condConvInB;        // conv_in 128→384 3×3
    private readonly ResNetBlock2D[] _condRes = new ResNetBlock2D[3];   // block.{0,2,4}
    private readonly VaeAttention[] _condAttn = new VaeAttention[2];    // block.{1,3}
    private Tensor? _condNormOutW, _condNormOutB;      // GroupNorm 384
    private Tensor? _condConvOutW, _condConvOutB;      // conv_out 384→384 3×3

    // ── s_embedder (proj1 skipped: zero-noise → 0) ──
    private Tensor? _sProj2W, _sProj2B;                // proj2 Conv1×1 512→384

    // ── DiCo trunk ──
    private readonly MageDiCoBlock[] _dico = new MageDiCoBlock[NumDiCoBlocks];
    private Tensor? _tEmbW1, _tEmbB1, _tEmbW2, _tEmbB2; // t_embedder MLP (256→384→384); evaluated once at t=0

    // ── NeRF per-patch decode ──
    private Tensor? _yEmbXW, _yEmbXB;                  // y_embedder_x Conv1×1 384→(32·256=8192)
    private Tensor? _xEmbW, _xEmbB;                    // x_embedder Linear (35+64=99 → 32)
    private Tensor? _decCondEmbW, _decCondEmbB;        // dec_net.cond_embed Linear 384→(16²·32=8192)
    private Tensor? _decInputProjW, _decInputProjB;    // dec_net.input_proj Linear 32→32
    private readonly MageMlpResBlock[] _decRes = new MageMlpResBlock[DecResBlocks];
    private Tensor? _finalNormW;                       // final_layer.norm RMSNorm 32
    private Tensor? _finalLinW, _finalLinB;            // final_layer.linear 32→3

    private float[]? _dctCoords;                       // fixed [256, 64] DCT coordinate table
    private int _disposed;

    public MageVaeDecoder()
    {
        for (int i = 0; i < 3; i++) _condRes[i] = new ResNetBlock2D(Hidden, Hidden, GroupNormGroups, GroupNormEps);
        for (int i = 0; i < 2; i++) _condAttn[i] = new VaeAttention(Hidden, GroupNormGroups, GroupNormEps);
        for (int i = 0; i < NumDiCoBlocks; i++) _dico[i] = new MageDiCoBlock(Hidden);
        for (int i = 0; i < DecResBlocks; i++) _decRes[i] = new MageMlpResBlock(HiddenX);
    }

    /// <summary>Loads decoder weights. All keys are under <c>pipeline.*</c> (the wrapper strips <c>pipeline.</c> and
    /// skips <c>y_embedder.encoder</c>/<c>y_embedder.bottleneck</c>). Pass the already-<c>pipeline.</c>-stripped dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // CoD decoder (y_embedder.decoder.*).
        _condConvInW = F32(w["y_embedder.decoder.conv_in.weight"]);
        _condConvInB = F32(w["y_embedder.decoder.conv_in.bias"]);
        _condRes[0].LoadWeights(w, "y_embedder.decoder.block.0");
        _condAttn[0].LoadWeights(w, "y_embedder.decoder.block.1");
        _condRes[1].LoadWeights(w, "y_embedder.decoder.block.2");
        _condAttn[1].LoadWeights(w, "y_embedder.decoder.block.3");
        _condRes[2].LoadWeights(w, "y_embedder.decoder.block.4");
        _condNormOutW = F32(w["y_embedder.decoder.norm_out.weight"]);
        _condNormOutB = F32(w["y_embedder.decoder.norm_out.bias"]);
        _condConvOutW = F32(w["y_embedder.decoder.conv_out.weight"]);
        _condConvOutB = F32(w["y_embedder.decoder.conv_out.bias"]);

        _sProj2W = F32(w["s_embedder.proj2.weight"]);
        _sProj2B = F32(w["s_embedder.proj2.bias"]);

        _tEmbW1 = F32(w["t_embedder.mlp.0.weight"]);
        _tEmbB1 = F32(w["t_embedder.mlp.0.bias"]);
        _tEmbW2 = F32(w["t_embedder.mlp.2.weight"]);
        _tEmbB2 = F32(w["t_embedder.mlp.2.bias"]);
        for (int i = 0; i < NumDiCoBlocks; i++) _dico[i].LoadWeights(w, $"blocks.{i}");

        _yEmbXW = F32(w["y_embedder_x.weight"]);
        _yEmbXB = F32(w["y_embedder_x.bias"]);
        _xEmbW = F32(w["x_embedder.embedder.0.weight"]);
        _xEmbB = F32(w["x_embedder.embedder.0.bias"]);
        _decCondEmbW = F32(w["dec_net.cond_embed.weight"]);
        _decCondEmbB = F32(w["dec_net.cond_embed.bias"]);
        _decInputProjW = F32(w["dec_net.input_proj.weight"]);
        _decInputProjB = F32(w["dec_net.input_proj.bias"]);
        for (int i = 0; i < DecResBlocks; i++) _decRes[i].LoadWeights(w, $"dec_net.res_blocks.{i}");
        _finalNormW = F32(w["final_layer.norm.weight"]);
        _finalLinW = F32(w["final_layer.linear.weight"]);
        _finalLinB = F32(w["final_layer.linear.bias"]);

        _dctCoords = BuildDctCoords();
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _condConvInW, _condConvInB, _condNormOutW, _condNormOutB, _condConvOutW,
            _condConvOutB, _sProj2W, _sProj2B, _tEmbW1, _tEmbB1, _tEmbW2, _tEmbB2, _yEmbXW, _yEmbXB, _xEmbW, _xEmbB,
            _decCondEmbW, _decCondEmbB, _decInputProjW, _decInputProjB, _finalNormW, _finalLinW, _finalLinB })
            if (t is not null) yield return t;
        foreach (ResNetBlock2D b in _condRes) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (VaeAttention b in _condAttn) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (MageDiCoBlock b in _dico) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (MageMlpResBlock b in _decRes) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes latent <c>[B, 128, h, w]</c> → RGB <c>[B, 3, H, W]</c> in <c>[-1, 1]</c>. Deterministic
    /// (t = 0, zero noise). Callers map to bytes as <c>((x.Clamp(-1,1)+1)*127.5).Byte()</c> (truncation).</summary>
    public Tensor Decode(IBackend backend, Tensor z)
    {
        int b = (int)z.Shape[0], h = (int)z.Shape[2], wpx = (int)z.Shape[3];
        int H = h * Patch, W = wpx * Patch, hw = h * wpx;

        // 1. CoD decoder: z → cond [b, 384, h, w].
        Tensor cond = DecodeCod(backend, z, b, h, wpx);

        // 2. s_embedder: proj1(0)=0 → s = proj2( concat([0(128) ; cond(384)]) ) [b, 384, h, w].
        Tensor sIn = new(new TensorShape(b, LatentChannels + Hidden, h, wpx), DType.F32);
        Zero(sIn);
        CopyChannels(cond, sIn, 0, LatentChannels, Hidden);   // cond → channels [128, 512)
        Tensor s = new(new TensorShape(b, Hidden, h, wpx), DType.F32);
        backend.Conv2D(s, sIn, _sProj2W!, _sProj2B, 1, 1, 0, 0);
        sIn.Dispose();

        // 3. DiCo trunk at latent res, modulated by the constant c = t_embedder(0).
        Tensor c = TimestepEmbedZero(backend, b);   // [b, 384]
        for (int i = 0; i < NumDiCoBlocks; i++)
        {
            Tensor next = _dico[i].Forward(backend, s, c);
            s.Dispose(); s = next;
        }
        c.Dispose();

        // 4. NeRF per-patch decode. Reference plumbing (mage_vae.py L499-513):
        //    s_tokens = s.permute(0,2,3,1).reshape(b*hw, 384)
        //    yx = y_embedder_x(cond).flatten(2)  → [b, 8192, hw]; x = cat([unfold(0)=0(768), yx], 1) = [b, 8960, hw]
        //    x = x.reshape(b,35,256,hw).permute(0,3,2,1).flatten(0,1) → [b*hw, 256, 35]
        //    x = x_embedder(x); x = dec_net(x, s_tokens); x = final_layer(x) → [b*hw, 256, 3]
        //    fold → [b, 3, H, W]
        Tensor sTokens = PermuteNchwToTokens(s, b, Hidden, h, wpx);   // [b*hw, 384]
        s.Dispose();

        Tensor yx = new(new TensorShape(b, HiddenX * SubPixels, h, wpx), DType.F32);   // [b, 8192, h, w]
        backend.Conv2D(yx, cond, _yEmbXW!, _yEmbXB, 1, 1, 0, 0);
        cond.Dispose();

        // Build the per-token NeRF input [b*hw, 256, 35]: first 3 sub-channels are the (zero) unfolded noise,
        // the remaining 32 come from yx reshaped as [b, 32, 256, hw]. See parity trap (5).
        Tensor nerfIn = BuildNerfInput(yx, b, h, wpx);   // [b*hw, 256, 35]
        yx.Dispose();

        Tensor xEmb = XEmbed(backend, nerfIn, b * hw);   // [b*hw, 256, 32]
        nerfIn.Dispose();

        Tensor decoded = DecNet(backend, xEmb, sTokens, b * hw);   // [b*hw, 256, 32]
        xEmb.Dispose(); sTokens.Dispose();

        Tensor rgbTokens = FinalLayer(backend, decoded, b * hw);   // [b*hw, 256, 3]
        decoded.Dispose();

        // 5. Fold the h·w tiles of 16×16×3 back into [b, 3, H, W].
        Tensor img = Fold(rgbTokens, b, h, wpx, H, W);
        rgbTokens.Dispose();
        return img;
    }

    // ── CoD decoder ──
    private Tensor DecodeCod(IBackend backend, Tensor z, int b, int h, int w)
    {
        Tensor x = new(new TensorShape(b, Hidden, h, w), DType.F32);
        backend.Conv2D(x, z, _condConvInW!, _condConvInB, 1, 1, 1, 1);   // conv_in 128→384 3×3
        Tensor r0 = _condRes[0].Forward(backend, x); x.Dispose();
        Tensor a0 = _condAttn[0].Forward(backend, r0); r0.Dispose();
        Tensor r1 = _condRes[1].Forward(backend, a0); a0.Dispose();
        Tensor a1 = _condAttn[1].Forward(backend, r1); r1.Dispose();
        Tensor r2 = _condRes[2].Forward(backend, a1); a1.Dispose();
        Tensor normed = new(r2.Shape, DType.F32);
        backend.GroupNormSilu(normed, r2, _condNormOutW!, _condNormOutB!, GroupNormGroups, GroupNormEps);
        r2.Dispose();
        Tensor outp = new(new TensorShape(b, Hidden, h, w), DType.F32);
        backend.Conv2D(outp, normed, _condConvOutW!, _condConvOutB, 1, 1, 1, 1);   // conv_out 384→384 3×3
        normed.Dispose();
        return outp;
    }

    // t_embedder(0): sinusoidal timestep_embedding(0, 256) = [1, 0, 0, …, cos(0)=1 …]. For t=0 the sinusoidal
    // embedding is [sin(0)=0 (×128) ; cos(0)=1 (×128)] → the "cos" half is all ones, "sin" half all zeros.
    private Tensor TimestepEmbedZero(IBackend backend, int b)
    {
        Tensor sinEmb = new(new TensorShape(b, 256), DType.F32);
        float* p = (float*)sinEmb.DataPointer;
        for (int i = 0; i < b; i++)
            for (int j = 0; j < 256; j++)
                p[i * 256 + j] = j >= 128 ? 1f : 0f;   // timestep_embedding(0): cat[sin(0), cos(0)]
        Tensor t1 = new(new TensorShape(b, Hidden), DType.F32);
        backend.Linear(t1, sinEmb, _tEmbW1!, _tEmbB1);
        sinEmb.Dispose();
        Tensor act = new(t1.Shape, DType.F32);
        backend.Silu(act, t1); t1.Dispose();
        Tensor c = new(new TensorShape(b, Hidden), DType.F32);
        backend.Linear(c, act, _tEmbW2!, _tEmbB2); act.Dispose();
        return c;
    }

    // x_embedder: for each of the b*hw*256 rows, concat the 35 input channels with the fixed 64 DCT coords → 99,
    // then Linear(99→32). The DCT table is the same 256×64 for every token.
    private Tensor XEmbed(IBackend backend, Tensor nerfIn, int rows)   // nerfIn [rows, 256, 35]
    {
        int inDim = 35 + MaxFreqs * MaxFreqs;   // 99
        Tensor cat = new(new TensorShape(rows * SubPixels, inDim), DType.F32);
        float* dst = (float*)cat.DataPointer;
        float* src = (float*)nerfIn.DataPointer;   // [rows, 256, 35]
        float* dct = fixedDct();
        for (long r = 0; r < rows; r++)
            for (int sp = 0; sp < SubPixels; sp++)
            {
                long o = (r * SubPixels + sp) * inDim;
                long si = (r * SubPixels + sp) * 35;
                for (int k = 0; k < 35; k++) dst[o + k] = src[si + k];
                for (int k = 0; k < 64; k++) dst[o + 35 + k] = dct[sp * 64 + k];
            }
        Tensor outp = new(new TensorShape(rows * SubPixels, HiddenX), DType.F32);
        backend.Linear(outp, cat, _xEmbW!, _xEmbB);
        cat.Dispose();
        return outp.Reshape(new TensorShape(rows, SubPixels, HiddenX));
    }

    // dec_net (SimpleMLPAdaLN): input_proj(32→32); c = cond_embed(s_tokens)[384→8192].reshape(rows,256,32);
    // then DecResBlocks × MLPResBlock(x, c).
    private Tensor DecNet(IBackend backend, Tensor x, Tensor sTokens, int rows)   // x [rows,256,32], sTokens [rows,384]
    {
        Tensor xFlat = x.Reshape(new TensorShape(rows * SubPixels, HiddenX));
        Tensor xp = new(new TensorShape(rows * SubPixels, HiddenX), DType.F32);
        backend.Linear(xp, xFlat, _decInputProjW!, _decInputProjB);
        Tensor cond = new(new TensorShape(rows, SubPixels * HiddenX), DType.F32);   // [rows, 8192]
        backend.Linear(cond, sTokens, _decCondEmbW!, _decCondEmbB);
        Tensor cRes = cond.Reshape(new TensorShape(rows, SubPixels, HiddenX));
        Tensor cur = xp.Reshape(new TensorShape(rows, SubPixels, HiddenX));
        for (int i = 0; i < DecResBlocks; i++)
        {
            Tensor next = _decRes[i].Forward(backend, cur, cRes, rows);
            cur.Dispose(); cur = next;
        }
        cond.Dispose();
        return cur;
    }

    // final_layer: RMSNorm(32) → Linear(32→3).
    private Tensor FinalLayer(IBackend backend, Tensor x, int rows)   // [rows,256,32] → [rows,256,3]
    {
        Tensor xFlat = x.Reshape(new TensorShape(rows * SubPixels, HiddenX));
        Tensor normed = new(xFlat.Shape, DType.F32);
        backend.RmsNorm(normed, xFlat, _finalNormW!, 1e-6f);
        Tensor rgb = new(new TensorShape(rows * SubPixels, 3), DType.F32);
        backend.Linear(rgb, normed, _finalLinW!, _finalLinB);
        normed.Dispose();
        return rgb.Reshape(new TensorShape(rows, SubPixels, 3));
    }

    // ── host-side tensor plumbing (small, one-time per decode; correctness over cleverness) ──

    private float* fixedDct() { fixed (float* p = _dctCoords!) return p; }

    // fetch_pos: fixed 256×64 DCT coordinate table (mage_vae.py NerfEmbedder.fetch_pos). Parity trap (1)/(2):
    // freqs = linspace(0,8,8) endpoint-inclusive; meshgrid(indexing="ij") → (pos_y, pos_x).
    private static float[] BuildDctCoords()
    {
        float[] pos = new float[Patch];
        for (int i = 0; i < Patch; i++) pos[i] = i / (float)(Patch - 1);   // linspace(0,1,16)
        float[] freqs = new float[MaxFreqs];
        for (int i = 0; i < MaxFreqs; i++) freqs[i] = MaxFreqs * i / (float)(MaxFreqs - 1);   // linspace(0,8,8)
        float[] table = new float[SubPixels * MaxFreqs * MaxFreqs];
        for (int py = 0; py < Patch; py++)
            for (int px = 0; px < Patch; px++)
            {
                int sp = py * Patch + px;   // meshgrid ij row-major flatten
                for (int fx = 0; fx < MaxFreqs; fx++)
                    for (int fy = 0; fy < MaxFreqs; fy++)
                    {
                        float coeff = 1f / (1f + freqs[fx] * freqs[fy]);
                        float dctX = MathF.Cos(pos[px] * freqs[fx] * MathF.PI);
                        float dctY = MathF.Cos(pos[py] * freqs[fy] * MathF.PI);
                        table[sp * 64 + fx * MaxFreqs + fy] = dctX * dctY * coeff;
                    }
            }
        return table;
    }

    // s.permute(0,2,3,1).reshape(b*h*w, C): NCHW → tokens.
    private static Tensor PermuteNchwToTokens(Tensor s, int b, int c, int h, int w)
    {
        int hw = h * w;
        Tensor outp = new(new TensorShape(b * hw, c), DType.F32);
        float* src = (float*)s.DataPointer;   // [b, c, h, w]
        float* dst = (float*)outp.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
                for (int ch = 0; ch < c; ch++)
                    dst[(bi * hw + p) * c + ch] = src[((bi * c + ch) * hw) + p];
        return outp;
    }

    // Build [b*hw, 256, 35]: rows 0..2 (rgb sub-channels) = 0 (zero noise); rows 3..34 = yx[b,32,256,hw].
    // Matches cat([unfold(0)=0, yx],1).reshape(b,35,256,hw).permute(0,3,2,1).flatten(0,1).  Parity trap (5)/(7).
    private static Tensor BuildNerfInput(Tensor yx, int b, int h, int w)   // yx [b, 8192, h, w] = [b, 32*256, hw]
    {
        int hw = h * w;
        Tensor outp = new(new TensorShape(b * hw, SubPixels, 35), DType.F32);
        Zero(outp);
        float* src = (float*)yx.DataPointer;   // [b, 32*256, hw] contiguous; channel index = emb*256 + sp
        float* dst = (float*)outp.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
                for (int sp = 0; sp < SubPixels; sp++)
                {
                    long o = ((long)(bi * hw + p) * SubPixels + sp) * 35;
                    // rows 0..2 stay 0; rows 3..34 = yx[bi, emb*256+sp, p]
                    for (int emb = 0; emb < HiddenX; emb++)
                        dst[o + 3 + emb] = src[((long)bi * (HiddenX * SubPixels) + (long)emb * SubPixels + sp) * hw + p];
                }
        return outp;
    }

    // fold(kernel=stride=16): rgbTokens [b*hw, 256, 3] → [b, 3, H, W]. Token p = (row,col) in the h×w grid; its
    // 256 sub-pixels tile a 16×16 block. Parity trap (6): PyTorch fold sub-pixel order is row-major (sp = ky*16+kx).
    private static Tensor Fold(Tensor rgb, int b, int h, int w, int H, int W)
    {
        Tensor img = new(new TensorShape(b, 3, H, W), DType.F32);
        float* src = (float*)rgb.DataPointer;   // [b*hw, 256, 3]
        float* dst = (float*)img.DataPointer;
        int hw = h * w;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
            {
                int trow = p / w, tcol = p - trow * w;
                for (int sp = 0; sp < SubPixels; sp++)
                {
                    int ky = sp / Patch, kx = sp - ky * Patch;
                    int y = trow * Patch + ky, x = tcol * Patch + kx;
                    long si = ((long)(bi * hw + p) * SubPixels + sp) * 3;
                    for (int ch = 0; ch < 3; ch++)
                        dst[((long)(bi * 3 + ch) * H + y) * W + x] = src[si + ch];
                }
            }
        return img;
    }

    private static void CopyChannels(Tensor src, Tensor dst, int srcC0, int dstC0, int count)
    {
        int b = (int)src.Shape[0], hw = (int)src.Shape[2] * (int)src.Shape[3];
        int srcCh = (int)src.Shape[1], dstCh = (int)dst.Shape[1];
        float* s = (float*)src.DataPointer; float* d = (float*)dst.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < count; c++)
                Buffer.MemoryCopy(s + ((long)(bi * srcCh + srcC0 + c) * hw), d + ((long)(bi * dstCh + dstC0 + c) * hw),
                    (long)hw * 4, (long)hw * 4);
    }

    private static void Zero(Tensor t) => new Span<float>((float*)t.DataPointer, (int)t.ElementCount).Clear();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Weight tensors are owned by the loaded checkpoint dict; nothing unmanaged to free here beyond letting GC
        // reclaim the F32 copies. Block helpers hold weight references only.
    }
}
