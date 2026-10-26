#!/usr/bin/env python3
"""Compare patched Rust decoders with independently compiled ggml C decoders."""
import argparse, array, ctypes as C, random, struct, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ggml-lib',required=True);p.add_argument('--probe',required=True);p.add_argument('--fixtures');a=p.parse_args()
lib=C.CDLL(a.ggml_lib)
rng=random.Random(20261023)
for ty,bs,at,name in [(10,84,80,'q2_K'),(18,98,0,'iq3_xxs')]:
    # Random code patterns plus scales spanning signs, zero, subnormal and normal values.
    nb=4096; raw=bytearray(rng.randbytes(nb*bs)); scales=[0.,2**-24,2**-14,-0.25,0.125,0.3,1.,3.,16.]
    for b in range(nb):
        struct.pack_into('<e',raw,b*bs+at,scales[b%len(scales)])
        if ty==10:struct.pack_into('<e',raw,b*bs+82,scales[(b+3)%len(scales)])
    fn=getattr(lib,'dequantize_row_'+name);fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int64];fn.restype=None
    src=(C.c_ubyte*len(raw)).from_buffer(raw);ref=(C.c_float*(nb*256))();fn(src,ref,nb*256)
    expected=C.string_at(ref,C.sizeof(ref))
    got=subprocess.run([a.probe,str(ty)],input=raw,stdout=subprocess.PIPE,check=True).stdout
    assert len(got)==len(expected)
    diff=sum(x!=y for x,y in zip(struct.iter_unpack('<f',got),struct.iter_unpack('<f',expected)))
    assert diff==0, (name,diff)
    print(f'{name}: {nb*256:,} weights match ggml exactly')
    if a.fixtures:
        dest=Path(a.fixtures);dest.mkdir(parents=True,exist_ok=True)
        # Small immutable subset for offline regression tests.
        (dest/(name+'.raw')).write_bytes(raw[:64*bs])
        (dest/(name+'.f32')).write_bytes(expected[:64*256*4])
