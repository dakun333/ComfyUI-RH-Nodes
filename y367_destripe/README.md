# Y367-FAST CF1 异常保护

1.9.1 仅保留 `Y367CF1ProtectedDestripe`（Y367-FAST CF1 异常保护），分类 `RH-Nodes/restore`。基础CF1、CE4和CE4异常保护已移除，旧节点工作流需手动换成此节点并重新连接三个输出。

保留CF1轮廓方向保护、ST1窄过渡和PO1局部异常回撤；不改变模型、颜色许可、五轮逻辑恢复或保留节点的ID/参数/输出。

Load Image IMAGE → image，透明图片另接MASK → mask，处理图像 → Save Image。MASK是透明度（1透明），不是去纹范围。orientation可选vertical/horizontal/both（默认both），固定48周期。输入为原生8位RGB/RGBA，拒绝任意FLOAT/HDR/16位；原尺寸与Alpha保留，Alpha<0.95的RGB原值保护。

ComfyUI V3输出为处理图像、删除量×16、报告。actual8（默认）适合标准Save Image；float32返回最终FLOAT，标准Save Image仍为8位、不等同冻结actual8编码。删除量×16以128灰为零，基于输入减所选输出。

CPU最多4核预算、批次逐帧，无独立环境/子进程。所选轴分析尺寸须大于120，首次执行包含Numba JIT。依赖见根requirements.txt，需要ComfyUI V3 API。

ST1仅对策略造成断点的允许侧做窄过渡；PO1对靠近保护区且幅度异常的已允许校正回撤，不增加CNN/FFT/恢复轮次。这是风险回撤，不是真实退化判定；可能少删近轮廓真纹、留下弱残留。IMAGE不传递ICC。

原始快照哈希保留于source_snapshot.json用于溯源。1.9.1删除其他节点及独立算法入口，将CF1需要的几何/整像素输出帮助函数收拢到contour_geometry.py。算法、封装与模型由贡献者授权MIT发布。
