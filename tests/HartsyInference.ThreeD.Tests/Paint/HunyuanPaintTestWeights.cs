using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Random weights in the Paint checkpoint's key layout for a tiny configuration, plus the same weights under the stock Stable Diffusion UNet's key names.</summary>
internal static unsafe class HunyuanPaintTestWeights
{
    internal static HunyuanPaintConfig TinyConfig(bool v21, int inChannels = 4) => new()
    {
        BlockOutChannels = [32, 32, 64, 64],
        Heads = [1, 1, 2, 2],
        LayersPerBlock = 1,
        CrossAttentionDim = 16,
        InChannels = inChannels,
        ReferenceInChannels = 4,
        TextTokens = 5,
        Materials = v21 ? ["albedo", "mr"] : ["albedo"],
        CameraClasses = v21 ? 0 : 7,
        UseDino = v21,
        UsePoseRope = v21,
    };

    /// <summary>Weights of one UNet stream. <paramref name="extras"/> adds the 2.5D tensors with random values scaled by <paramref name="extraScale"/>.</summary>
    internal static Dictionary<string, Tensor> Create(HunyuanPaintConfig cfg, bool generating, int seed, float extraScale = 1f)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> w = [];
        void Add(string key, float scale, params long[] shape)
        {
            int count = (int)shape.Aggregate(1L, (a, b) => a * b);
            Tensor t = new(new TensorShape(shape), DType.F32);
            for (int i = 0; i < count; i++)
                ((float*)t.DataPointer)[i] = (float)(random.NextDouble() * 2 - 1) * scale;
            w[key] = t;
        }
        void Norm(string key, int c)
        {
            Add(key + ".weight", 0.2f, c);
            for (int i = 0; i < c; i++)
                ((float*)w[key + ".weight"].DataPointer)[i] += 1f;
            Add(key + ".bias", 0.1f, c);
        }
        void Linear(string key, int outDim, int inDim, bool bias, float scale = 0.15f)
        {
            Add(key + ".weight", scale, outDim, inDim);
            if (bias)
                Add(key + ".bias", scale == 0f ? 0f : 0.05f, outDim);
        }
        void Attn(string key, int c, int kv, bool bias, float scale = 0.15f)
        {
            Linear(key + ".to_q", c, c, false, scale);
            Linear(key + ".to_k", c, kv, false, scale);
            Linear(key + ".to_v", c, kv, false, scale);
            Linear(key + ".to_out.0", c, c, true, scale);
        }
        void Resnet(string p, int cin, int cout, int timeDim)
        {
            Norm($"{p}.norm1", cin);
            Add($"{p}.conv1.weight", 0.08f, cout, cin, 3, 3); Add($"{p}.conv1.bias", 0.05f, cout);
            Linear($"{p}.time_emb_proj", cout, timeDim, true, 0.1f);
            Norm($"{p}.norm2", cout);
            Add($"{p}.conv2.weight", 0.08f, cout, cout, 3, 3); Add($"{p}.conv2.bias", 0.05f, cout);
            if (cin != cout)
            {
                Add($"{p}.conv_shortcut.weight", 0.1f, cout, cin, 1, 1); Add($"{p}.conv_shortcut.bias", 0.05f, cout);
            }
        }
        void Transformer(string p, int c)
        {
            Norm($"{p}.norm", c);
            Linear($"{p}.proj_in", c, c, true);
            Linear($"{p}.proj_out", c, c, true);
            string t = $"{p}.transformer_blocks.0.transformer";
            Norm($"{t}.norm1", c); Norm($"{t}.norm2", c); Norm($"{t}.norm3", c);
            Attn($"{t}.attn1", c, c, false);
            Attn($"{t}.attn2", c, cfg.CrossAttentionDim, false);
            Linear($"{t}.ff.net.0.proj", c * 8, c, true);
            Linear($"{t}.ff.net.2", c, c * 4, true);
            if (!generating)
                return;
            string b = $"{p}.transformer_blocks.0";
            Attn($"{b}.attn_multiview", c, c, true, 0.15f * extraScale);
            Attn($"{b}.attn_refview", c, c, true, 0.15f * extraScale);
            if (cfg.UseDino)
                Attn($"{b}.attn_dino", c, cfg.CrossAttentionDim, true, 0.15f * extraScale);
            for (int m = 1; m < cfg.MaterialCount; m++)
            {
                string mat = cfg.Materials[m];
                foreach (string part in new[] { "q", "k", "v" })
                    Linear($"{t}.attn1.processor.to_{part}_{mat}", c, c, false);
                Linear($"{t}.attn1.processor.to_out_{mat}.0", c, c, true);
                Linear($"{b}.attn_refview.processor.to_v_{mat}", c, c, false, 0.15f * extraScale);
                Linear($"{b}.attn_refview.processor.to_out_{mat}.0", c, c, true, 0.15f * extraScale);
            }
        }

        int[] ch = cfg.BlockOutChannels;
        int timeDim = cfg.TimeEmbeddingDim, levels = ch.Length, layers = cfg.LayersPerBlock;
        Add("conv_in.weight", 0.1f, ch[0], generating ? cfg.InChannels : cfg.ReferenceInChannels, 3, 3); Add("conv_in.bias", 0.05f, ch[0]);
        Linear("time_embedding.linear_1", timeDim, ch[0], true, 0.1f);
        Linear("time_embedding.linear_2", timeDim, timeDim, true, 0.1f);
        if (cfg.CameraClasses > 0)
            Add("class_embedding.weight", 0.3f, cfg.CameraClasses, timeDim);
        for (int i = 0; i < levels; i++)
        {
            for (int j = 0; j < layers; j++)
            {
                Resnet($"down_blocks.{i}.resnets.{j}", j == 0 ? (i == 0 ? ch[0] : ch[i - 1]) : ch[i], ch[i], timeDim);
                if (i < levels - 1)
                    Transformer($"down_blocks.{i}.attentions.{j}", ch[i]);
            }
            if (i < levels - 1)
            {
                Add($"down_blocks.{i}.downsamplers.0.conv.weight", 0.08f, ch[i], ch[i], 3, 3); Add($"down_blocks.{i}.downsamplers.0.conv.bias", 0.05f, ch[i]);
            }
        }
        Resnet("mid_block.resnets.0", ch[^1], ch[^1], timeDim);
        Transformer("mid_block.attentions.0", ch[^1]);
        Resnet("mid_block.resnets.1", ch[^1], ch[^1], timeDim);
        for (int i = 0; i < levels; i++)
        {
            int level = levels - 1 - i, outCh = ch[level], prevOut = i == 0 ? ch[^1] : ch[level + 1], inputCh = ch[Math.Max(level - 1, 0)];
            for (int j = 0; j <= layers; j++)
            {
                Resnet($"up_blocks.{i}.resnets.{j}", (j == 0 ? prevOut : outCh) + (j == layers ? inputCh : outCh), outCh, timeDim);
                if (i > 0)
                    Transformer($"up_blocks.{i}.attentions.{j}", outCh);
            }
            if (i < levels - 1)
            {
                Add($"up_blocks.{i}.upsamplers.0.conv.weight", 0.08f, outCh, outCh, 3, 3); Add($"up_blocks.{i}.upsamplers.0.conv.bias", 0.05f, outCh);
            }
        }
        Norm("conv_norm_out", ch[0]);
        Add("conv_out.weight", 0.1f, cfg.OutChannels, ch[0], 3, 3); Add("conv_out.bias", 0.05f, cfg.OutChannels);
        return w;
    }

    /// <summary>The generating stream's tensors under the stock UNet's names: the wrapped transformer's segment is dropped and the 2.5D extras are left out.</summary>
    internal static Dictionary<string, Tensor> StockNames(Dictionary<string, Tensor> paint)
    {
        Dictionary<string, Tensor> stock = [];
        foreach ((string key, Tensor tensor) in paint)
        {
            if (key.Contains(".attn_") || key.Contains(".processor."))
                continue;
            stock[key.Replace(".transformer_blocks.0.transformer.", ".transformer_blocks.0.")] = tensor;
        }
        return stock;
    }
}
