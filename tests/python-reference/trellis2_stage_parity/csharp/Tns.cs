using System.Text;

/// <summary>Tiny tensor container shared with the Python side: repeated [int32 nameLen][name][int32 dtype 0=f32 1=i32][int32 ndim][int64 dims...][data].</summary>
static class Tns
{
    public static void Write(string path, params (string Name, Array Data, long[] Dims)[] items)
    {
        using BinaryWriter w = new(File.Create(path));
        foreach ((string name, Array data, long[] dims) in items)
        {
            byte[] n = Encoding.UTF8.GetBytes(name);
            w.Write(n.Length); w.Write(n);
            int dtype = data is float[] ? 0 : data is int[] ? 1 : throw new ArgumentException("float[] or int[] only");
            w.Write(dtype); w.Write(dims.Length);
            foreach (long d in dims) w.Write(d);
            byte[] raw = new byte[Buffer.ByteLength(data)];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            w.Write(raw);
        }
    }

    public static Dictionary<string, (Array Data, long[] Dims)> Read(string path)
    {
        Dictionary<string, (Array, long[])> result = new();
        using BinaryReader r = new(File.OpenRead(path));
        while (r.BaseStream.Position < r.BaseStream.Length)
        {
            string name = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
            int dtype = r.ReadInt32(), nd = r.ReadInt32();
            long[] dims = new long[nd]; long count = 1;
            for (int i = 0; i < nd; i++) { dims[i] = r.ReadInt64(); count *= dims[i]; }
            Array data = dtype == 0 ? new float[count] : new int[count];
            byte[] raw = r.ReadBytes((int)(count * 4));
            Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
            result[name] = (data, dims);
        }
        return result;
    }
}
