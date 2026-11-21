using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.Core.Runtime;

/// <summary>Moves a finished temporary file over its target in one step, so a reader sees the old file or the new one.</summary>
/// <remarks>On Windows <c>File.Move</c> with overwrite is refused while any handle is open on the target; the rename
/// here replaces it under handles opened with <see cref="FileShare.Delete"/>, as <c>rename(2)</c> does elsewhere.</remarks>
internal static partial class AtomicFileReplace
{
    private const uint DeleteAccess = 0x00010000;
    private const uint SynchronizeAccess = 0x00100000;
    private const int FileRenameInfoEx = 22;
    private const uint RenameReplaceIfExists = 0x1;
    private const uint RenamePosixSemantics = 0x2;

    /// <summary>Byte offset of <c>FILE_RENAME_INFO.FileName</c> on 64-bit Windows, after the flags, root handle and name length.</summary>
    private const int RenameInfoNameOffset = 20;

    /// <summary>Replaces <paramref name="path"/> with <paramref name="temp"/> (same volume); throws when the target cannot be replaced.</summary>
    public static void Move(string temp, string path)
    {
        if (OperatingSystem.IsWindows() && Environment.Is64BitProcess && TryPosixRename(temp, Path.GetFullPath(path))) return;
        // 32-bit processes and volumes without POSIX rename land here, where an open handle on the target makes it throw.
        File.Move(temp, path, overwrite: true);
    }

    [SupportedOSPlatform("windows")]
    private static unsafe bool TryPosixRename(string temp, string fullPath)
    {
        using SafeFileHandle handle = CreateFile(temp, DeleteAccess | SynchronizeAccess,
            FileShare.ReadWrite | FileShare.Delete, 0, FileMode.Open, 0, 0);
        if (handle.IsInvalid) return false;

        int nameBytes = fullPath.Length * sizeof(char);
        byte[] info = new byte[RenameInfoNameOffset + nameBytes + sizeof(char)];
        fixed (byte* p = info)
        {
            *(uint*)p = RenameReplaceIfExists | RenamePosixSemantics;
            *(uint*)(p + 16) = (uint)nameBytes;
            MemoryMarshal.AsBytes(fullPath.AsSpan()).CopyTo(info.AsSpan(RenameInfoNameOffset));
            return SetFileInformationByHandle(handle, FileRenameInfoEx, p, (uint)info.Length);
        }
    }

    [LibraryImport("kernel32.dll", EntryPoint = "CreateFileW", SetLastError = true, StringMarshalling = StringMarshalling.Utf16)]
    private static partial SafeFileHandle CreateFile(string name, uint access, FileShare share, nint security,
        FileMode disposition, uint flags, nint template);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static unsafe partial bool SetFileInformationByHandle(SafeFileHandle handle, int infoClass, byte* info, uint size);
}
