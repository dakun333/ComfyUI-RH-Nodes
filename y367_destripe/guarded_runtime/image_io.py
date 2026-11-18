"""Native triplet IO and honest signed 8-bit removal (before - actual after)."""
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image

def read_image(path):
    with Image.open(path) as im:
        if im.mode not in ('RGB','RGBA'):raise ValueError(f'Expected RGB/RGBA, got {im.mode}')
        mode=im.mode;q=np.array(im.convert('RGBA'),np.uint8)
    return q[...,:3].astype(np.float32),q[...,3].astype(np.float32)/255,q[...,3].copy(),mode

def save_result(path,folder,rgb,alpha,alpha8,mode,target,eight,report,*,save_float=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(path,folder/'before.png')
    after=np.dstack((eight,alpha8)) if mode=='RGBA' else eight
    Image.fromarray(after).save(folder/'after.png')
    diff=rgb-eight.astype(np.float32);diag=np.uint8(np.clip(np.rint(128+16*diff),0,255))
    Image.fromarray(diag).save(folder/'removed_x16.png')
    reread=np.array(Image.open(folder/'after.png').convert('RGBA'),np.uint8)
    reread_diff=np.array(Image.open(folder/'removed_x16.png'),np.uint8)
    assert np.array_equal(reread[...,:3],eight) and np.array_equal(reread[...,3],alpha8) and np.array_equal(reread_diff,diag)
    assert np.array_equal(eight[alpha<.95],np.uint8(rgb[alpha<.95]))
    assert np.max(np.abs(eight.astype(np.float32)-target))<1.00001
    removal=rgb-target
    stats=dict(size=[rgb.shape[1],rgb.shape[0]],total_FLOAT_RMS=float(np.sqrt(np.mean(removal.astype(np.float64)**2))),total_actual8_RMS=float(np.sqrt(np.mean(diff.astype(np.float64)**2))),max_FLOAT_change=float(np.max(np.abs(removal))),max_actual8_change=int(np.max(np.abs(diff))),changed8_pixel_pct=float(np.mean(np.any(diff!=0,axis=2))*100),FLOAT8_error_max=float(np.max(np.abs(eight.astype(np.float32)-target))),alpha_exact=True,protected_RGB_exact=True,removed_png_is_actual8_signed_x16=True,native_size_preserved=True,native_png_reread=True)
    hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['before.png','after.png','removed_x16.png']}
    if save_float:np.savez_compressed(folder/'removed_FLOAT.npz',removed=removal)
    result=dict(before=str(folder/'before.png'),after=str(folder/'after.png'),removed_x16=str(folder/'removed_x16.png'),float=str(folder/'removed_FLOAT.npz') if save_float else None,stats=stats,report=report,png_sha256=hashes,input_file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
