# Y367-FAST 周期去纹

在同一ComfyUI Python和一套依赖中运行，纯CPU；不创建独立环境或子进程。

| 节点 | 保护方式 |
|---|---|
| Y367-FAST CF1 | 按原图轮廓方向保护 |
| Y367-FAST CE4 | 保护强边缘及附近区域 |
| Y367-FAST CF1 异常保护 | CF1 + ST1窄过渡 + PO1局部异常回撤 |
| Y367-FAST CE4 异常保护 | CE4 + ST1窄过渡 + PO1局部异常回撤 |

原始未加保护的`Y367FastDestripe`已移除，不保留注册别名。现有CF1/CE4名称、接口及输出行为不变。

## 使用

`Load Image → 去纹节点 → Save Image`。连接透明图片的Load Image MASK；MASK是透明度，不是去纹范围。选择vertical（竖纹）、horizontal（横纹）或both（联合）。固定48周期、每轴五轮逻辑更新，精确固定点复用不减少逻辑轮次。

默认actual8适合标准Save Image。float32返回最终FLOAT，标准Save Image仍会保存8位，不能等同于本版定向编码。输入须为原生8位RGB/RGBA；不静默量化任意FLOAT。尺寸与Alpha保持，IMAGE不传递原ICC元数据。

新异常保护节点输出处理图像、删除量×16和报告；Alpha随处理图像保留，不额外输出透传MASK。现有CF1/CE4仍保留原四个输出，兼容已有连线。

删除量×16以128灰为零，来自输入RGB减实际8位输出（actual8）或最终FLOAT（float32），不做饱和度增强。

## ST1 + PO1

ST1只在策略造成的断点允许侧做窄过渡，宽度为短边1024像素时6像素。PO1仅在靠近原保护区且幅度超过局部参考约2倍时额外回撤；不扩大原保护区、不全图限幅、不向原零删除区扩散。FLOAT校正保持整份RGB方向，实际8位各通道不比ST1更激进。

检测模型、颜色许可及五轮处理不变，新增步骤不增加CNN、FFT或恢复轮次。较强PO2不注册、不开放。

这属于风险回撤，不是误删真值判定；近轮廓的强真实纹理可能少删，弱尖峰和旧残留也可能保留。首次执行包含Numba JIT，热耗时不代表冷启动保证。

RH接入使用仓库根`requirements.txt`，原始文件哈希见`source_snapshot.json`。保留运行核心和模型，分类调整为`RH-Nodes/restore`；异常保护版需ComfyUI V3 API。贡献者确认算法、封装及权重为原创并授权MIT发布。源码快照来源为未独立Git跟踪的本地Y367-FAST目录。
