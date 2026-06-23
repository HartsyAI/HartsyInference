import sys; sys.modules['torchvision']=None
import numpy as np, torch
import torchaudio.compliance.kaldi as k
# deterministic signal in [-1,1], 1.0s @ 16k
n=16000
t=np.arange(n)/16000.0
x=(0.6*np.sin(2*np.pi*220*t)+0.3*np.sin(2*np.pi*440*t)+0.1*np.sin(2*np.pi*1500*t)).astype(np.float32)
wav=torch.from_numpy(x).unsqueeze(0)
feat=k.fbank(wav, num_mel_bins=80, dither=0.0, sample_frequency=16000)  # [T,80], defaults
arr=feat.numpy().astype(np.float32)
np.save("/tmp/kaldi_ref.npy", arr)
# also dump the input so C# reads identical samples
x.tofile("/tmp/kaldi_in.f32")
print("frames,bins", arr.shape, "mean", float(arr.mean()), "min", float(arr.min()), "max", float(arr.max()))
