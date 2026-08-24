using System.Collections.Generic;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Loads a CLIP textual-inversion embedding (a learned <c>&lt;embed:name&gt;</c> token) from a <c>.safetensors</c> or <c>.pt</c> file into a <c>[numVectors, hiddenSize]</c> F32 tensor. Handles the common layouts (A1111 <c>string_to_param['*']</c>, diffusers <c>emb_params</c>, SDXL dual <c>clip_l</c>/<c>clip_g</c>) by selecting the 2D tensor whose feature dim matches the target encoder; the caller assigns placeholder token ids to each row and passes them to <c>ClipTextEncoder.Encode(..., inlineEmbeddings)</c>.</summary>
public static class TextualInversion
{
    /// <summary>Loads the embedding matching <paramref name="hiddenSize"/> from <paramref name="path"/>. Returns <c>[numVectors, hiddenSize]</c> F32.</summary>
    public static Tensor Load(string path, int hiddenSize)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Textual-inversion file not found: {path}");
        }
        // SelectEmbedding MUST run while the loader is still alive: the tensors it returns borrow the loader's
        // mmap, so copying out of them after the loader's `using` disposes would read freed memory (AccessViolation).
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            using SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            return SelectEmbedding(loader.GetAllTensors(), hiddenSize, path);
        }
        using PytorchPickleLoader pickleLoader = new PytorchPickleLoader();
        pickleLoader.Load(path);
        return SelectEmbedding(pickleLoader.GetAllTensors(), hiddenSize, path);
    }

    /// <summary>Slices a loaded <c>[numVectors, hiddenSize]</c> embedding into per-row inline-embedding entries keyed by sequential placeholder token ids starting at <paramref name="startPlaceholderId"/> (choose a value past the tokenizer vocab). The returned row tensors borrow <paramref name="embedding"/>'s memory, so keep it alive while encoding. Pass the map as <c>ClipTextEncoder.Encode(..., inlineEmbeddings)</c> and inject the ids where <c>&lt;embed:name&gt;</c> appears.</summary>
    public static unsafe (Dictionary<int, Tensor> Map, int[] PlaceholderIds) BuildInlineMap(Tensor embedding, int startPlaceholderId)
    {
        if (embedding.Shape.Rank != 2)
        {
            throw new HartsyInferenceException($"Embedding must be [numVectors, hidden]; got {embedding.Shape}.");
        }
        int numVectors = (int)embedding.Shape[0];
        int hidden = (int)embedding.Shape[1];
        Dictionary<int, Tensor> map = new Dictionary<int, Tensor>(numVectors);
        int[] ids = new int[numVectors];
        float* basePtr = (float*)embedding.DataPointer;
        for (int r = 0; r < numVectors; r++)
        {
            int id = startPlaceholderId + r;
            Tensor row = new Tensor(basePtr + (long)r * hidden, new TensorShape(hidden), embedding.DType);
            map[id] = row;
            ids[r] = id;
        }
        return (map, ids);
    }

    private static unsafe Tensor SelectEmbedding(Dictionary<string, Tensor> tensors, int hiddenSize, string path)
    {
        Tensor? chosen = null;
        foreach (KeyValuePair<string, Tensor> kv in tensors)
        {
            Tensor t = kv.Value;
            bool match = chosen is null && ((t.Shape.Rank == 2 && (int)t.Shape[1] == hiddenSize)
                || (t.Shape.Rank == 1 && (int)t.Shape[0] == hiddenSize));
            if (match)
            {
                chosen = t;
            }
            // Do NOT dispose the non-chosen tensors here: a multi-tensor embed (e.g. SDXL dual clip_l/clip_g)
            // has every tensor borrowing the SAME loader mmap, and disposing any one invalidates the shared
            // handle → the chosen tensor's pointer dangles → AccessViolation on the copy below. The loader's own
            // Dispose (its `using` in Load) tears the mmap down exactly once after this returns.
        }
        if (chosen is null)
        {
            throw new HartsyInferenceException($"No [N, {hiddenSize}] embedding found in {path}. Check the file matches the encoder's hidden size.");
        }
        // The loader's tensors borrow mmap memory that is freed when the loader disposes, so copy
        // into an owned [N, hidden] F32 tensor while the loader is still alive.
        int numVectors = chosen.Shape.Rank == 2 ? (int)chosen.Shape[0] : 1;
        Tensor srcF32 = chosen.DType == DType.F32 ? chosen : chosen.CastTo(DType.F32);
        Tensor result = new Tensor(new TensorShape(numVectors, hiddenSize), DType.F32);
        long bytes = (long)numVectors * hiddenSize * sizeof(float);
        Buffer.MemoryCopy((void*)srcF32.DataPointer, (void*)result.DataPointer, bytes, bytes);
        if (!ReferenceEquals(srcF32, chosen))
        {
            srcF32.Dispose();
        }
        return result;
    }
}
