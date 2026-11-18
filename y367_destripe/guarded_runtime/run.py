"""Standalone CE4/CF1 CPU4: optimize execution, keep model and deletion policy."""
import os
import argparse, hashlib, json, time
from pathlib import Path
from PIL import Image
from .contour_edge import process as cf_process
from .edge_output import process as ce_process
from .residual_operator import MODEL_SHA256
from .image_io import read_image, save_result
ROOT=Path(__file__).absolute().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--method',choices=['CE4','CF1'],default='CF1')
    parser.add_argument('--orientation',choices=['both','vertical','horizontal'],default='both')
    parser.add_argument('--keep-edge-protection',action='store_true')
    parser.add_argument('--save-float',action='store_true')
    parser.add_argument('--no-fixed-point-reuse',action='store_true')
    parser.add_argument('--no-iteration-fusion',action='store_true')
    parser.add_argument('--cpu3-execution',action='store_true',help='Disable CPU4-only execution optimizations')
    parser.add_argument('--kernel-threads',type=int,choices=(1,2,4),default=None)
    parser.add_argument('--opencv-threads',type=int,choices=(1,2,4),default=None)
    args=parser.parse_args()
    from . import br10_core as core
    core.FIXED_POINT_REUSE=not args.no_fixed_point_reuse
    core.ITERATION_FUSION=not args.no_iteration_fusion
    from . import cpu_runtime
    import cv2
    from . import envelope as e
    if args.cpu3_execution:
        for name in ('FFT_LAYOUT','ANALYSIS_REUSE','HOT_KERNELS','CARRIER_FUSION','INVERSE_BATCH','TRANSFORM_REUSE'):setattr(cpu_runtime,name,False)
        if args.opencv_threads is not None:parser.error('--cpu3-execution cannot also set --opencv-threads')
        cv2.setNumThreads(1)
    if args.opencv_threads is not None:cv2.setNumThreads(args.opencv_threads)
    if args.kernel_threads is not None and e._hot is not None:
        try:e._hot.configure_threads(args.kernel_threads)
        except ValueError as exc:parser.error(str(exc))
    source=Path(args.input).absolute();destination=Path(args.output).absolute()
    if destination.exists():parser.error('Output already exists; refusing overwrite')
    paths=sorted(source.glob('*.png')) if source.is_dir() else [source]
    if not paths or not all(path.is_file() for path in paths):parser.error('Missing input PNG')
    if hashlib.sha256((ROOT/'model.npz').read_bytes()).hexdigest()!=MODEL_SHA256:
        parser.error('Frozen model hash changed')
    rows=[]
    for path in paths:
        started=time.perf_counter()
        with path.open('rb') as stream:header=stream.read(29)
        if len(header)!=29 or header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR' or header[24]!=8 or header[25] not in (2,6):
            raise ValueError('Only static native RGB/RGBA 8-bit PNG')
        with Image.open(path) as im:
            if getattr(im,'n_frames',1)!=1 or any(k in im.info for k in ('icc_profile','gamma','srgb','transparency')):
                raise ValueError('Unvalidated color-managed PNG; original input policy retained')
        rgb,alpha,alpha8,mode=read_image(path)
        target,eight,report=(ce_process if args.method=='CE4' else cf_process)(rgb,alpha,orientation=args.orientation,keep_edge_protection=args.keep_edge_protection)
        report.update(performance_implementation=args.method+'_CPU4_HOT',baseline='CPU3',detector_and_response_model_and_logical_five_passes_unchanged=True,CPU3_exact_state_reuse=core.FIXED_POINT_REUSE,CPU3_iteration_fusion=core.ITERATION_FUSION,CPU4_execution={name:getattr(cpu_runtime,name) for name in ('FFT_LAYOUT','ANALYSIS_REUSE','HOT_KERNELS','CARRIER_FUSION','INVERSE_BATCH','TRANSFORM_REUSE')},opencv_threads=cv2.getNumThreads(),native_kernel_threads=e._hot.configure_threads() if e._hot is not None else 0,CPU_only=True,quality_policy_unchanged=True)
        folder=destination/path.stem if source.is_dir() else destination
        record=save_result(path,folder,rgb,alpha,alpha8,mode,target,eight,report,save_float=args.save_float)
        record['read_process_encode_triplet_seconds']=time.perf_counter()-started
        (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append(record)
        print(json.dumps({'input':str(path),'output':str(folder),'seconds':record['read_process_encode_triplet_seconds'],'method':args.method+'_CPU4_HOT'},ensure_ascii=True),flush=True)
    if source.is_dir():(destination/'run.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()

