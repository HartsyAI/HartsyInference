"""RMBG-1.4 (BriaRMBG / ISNet) parity oracle. Loads briarmbg.py standalone, runs on a test image, dumps the
preprocessed input [1,3,1024,1024] + the raw sigmoid(d1) mask [1,1,1024,1024] (pre-postprocess). Also extracts
the weights (F32) for the C# loader. Env RMBG_IMAGE (default /tmp/TripoSR/examples/chair.png)."""
import os, sys, types, numpy as np, torch, torch.nn.functional as F, importlib.util as u
from safetensors.torch import save_file
from safetensors import safe_open
from PIL import Image
RMBG=os.environ.get("RMBG_DIR","/tmp/rmbg"); OUT=os.path.dirname(os.path.abspath(__file__))
IMG=os.environ.get("RMBG_IMAGE","/tmp/TripoSR/examples/chair.png")
# stub the transformers base + config so briarmbg.py imports standalone
tr=types.ModuleType("transformers")
tr.PreTrainedModel=type("PreTrainedModel",(torch.nn.Module,),{"__init__":lambda self,*a,**k: torch.nn.Module.__init__(self)})
sys.modules["transformers"]=tr
mc=types.ModuleType("MyConfig"); mc.RMBGConfig=type("RMBGConfig",(),{"__init__":lambda self,**k:None,"in_ch":3,"out_ch":1}); sys.modules["MyConfig"]=mc
spec=u.spec_from_file_location("briarmbg", os.path.join(RMBG,"briarmbg.py")); m=u.module_from_spec(spec)
# briarmbg does `from .MyConfig import RMBGConfig` — patch to absolute
src=open(os.path.join(RMBG,"briarmbg.py")).read().replace("from .MyConfig import RMBGConfig","from MyConfig import RMBGConfig")
exec(compile(src,"briarmbg.py","exec"), m.__dict__)
net=m.BriaRMBG(mc.RMBGConfig()); net.eval()
# load weights (safetensors → F32)
w={}
with safe_open(os.path.join(RMBG,"model.safetensors"),"pt") as f:
    for k in f.keys(): w[k]=f.get_tensor(k).float()
save_file(w, os.path.join(OUT,"rmbg_weights.safetensors")); print("weights:", len(w))
net.load_state_dict(w, strict=True)
# preprocess exactly like utilities.preprocess_image
im=np.array(Image.open(IMG).convert("RGB"))
t=torch.tensor(im,dtype=torch.float32).permute(2,0,1)
t=F.interpolate(t.unsqueeze(0),size=[1024,1024],mode='bilinear').type(torch.uint8)
image=(t/255.0 - 0.5)/1.0
with torch.no_grad():
    out=net(image)  # ([sig(d1..d6)],[feats])
    mask=out[0][0]   # sigmoid(d1) [1,1,1024,1024]
save_file({"input":image.contiguous(),"mask":mask.float().contiguous()}, os.path.join(OUT,"rmbg_ref_io.safetensors"))
print("input",tuple(image.shape),"mask",tuple(mask.shape),"mask min",mask.min().item(),"max",mask.max().item(),"mean",mask.mean().item())
