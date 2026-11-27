# VRMBG3-V6：私有模型时序去背

RH-nodes独立模块 `vrmbg3_v6/`，分类 `RH-Nodes/matting`，保留五个节点和ID：

| ID | 用途 |
| --- | --- |
| VRMBG3V6Loader | 加载模型，默认381对局部光晕数据训练ep4 / step71232 |
| VRMBG3V6Matting | IMAGE批次时序去背 |
| VRMBG3V6MattingVideo | 原生Load Video输入，传出帧率和音频 |
| VRMBG3V6LoadVideo | 旧式加载视频帧，限4GiB IMAGE |
| VRMBG3V6SaveVideo | 保存透明MOV、棋盘预览、可选16位Alpha |

## 模型单独提供

本仓库与wheel均不包含权重，不使用LFS，不自动从HF下载私有模型。将合法取得的权重单独放入ComfyUI标准 `models/checkpoints/`，在专用VRMBG3加载节点选择；普通SD检查点加载器不能加载它。

- 默认：`localhalo381_epoch004_s000071232.safetensors`，884915952字节；SHA-256 `49cc2001d47d73a3f06000e5a0c39e5a8deed6d5175e80278390199b62c0da06`。
- 兼容旧：`models/VRMBG3/V6_ep2/v6_epoch002_s000037440.safetensors`，884915840字节；SHA-256 `ed4e58962171ce1043cbc6ee1b7b2e72a1cfe20e7c6ac383d171c40e47640144`。仍兼容模块内models路径，额外模型路径可通过ComfyUI配置登记。
- strict校验模型身份及固定BRIA结构源码，不接受任意同名模型；旧工作流不会静默改用新checkpoint。

## 使用

原生Load Video VIDEO + VRMBG3模型加载 → VRMBG3原生VIDEO去背 → rgba/alpha/fps/audio/run_info → MOV保存。示例在example_workflows/；导入后选择自己的媒体和checkpoint。重启后端并刷新浏览器加载预览JS。

IMAGE批次是一段连续序列，每次调用重置历史；无关图片关闭temporal。输出原生尺寸，MASK为1前景、0背景，不做minmax、阈值、gamma；模型等比缩小及复制padding后的像素预算至多1048576，每边内容至少64。默认FP32权重、BF16前向、FP32 sigmoid，精度和offload接口保留，OOM提示降低max_pixels。

alpha_shrink/alpha_feather默认0/0，按原生像素执行连续灰度腐蚀再高斯羽化，同步修改Mask/RGBA/棋盘；不改变前景RGB或预测历史。enable_alpha_unmult默认False，关闭时RGB保留原输入；开启后multilevel（默认）使用PyMatting邻域前景/背景估计，legacy_q保留原Mask-Guided Q薄边保护。颜色恢复用最终Alpha，但历史仍使用原始预测；不保证复杂低Alpha颜色真值。

## 视频内存与保存

文件VIDEO的RGB批次超过1GiB时逐帧解码到FP32磁盘映射，共享RGB/RGBA存储；全部帧保持同一次时序调用。RGBA+棋盘约帧数×宽×高×28字节，另需2GiB临时盘余量，映射页仍占RAM。Windows稀疏文件和delete-on-close保留，结果/view引用释放后回收；音频超过10分钟需先Video Slice。文件型VFR拒绝，先转CFR。

- transparent.mov：默认ProRes4444，编码量化不是浮点无损；qtrle_lossless保留量化后的8位RGBA。
- checker.mp4：棋盘仅供预览，不是透明交付。
- alpha16.mkv：可选FFV1，16位Alpha/65535。
- metadata.json：模型、帧率、尺寸、音频尾部裁剪/补零及编码状态。

音频通过AUDIO显式连接，MOV为PCM16、预览AAC；保持速度，按帧数/fps补零或裁尾，不是原码流复制。预览JS在RH-nodes根web/，仅绑定保存节点，通过ComfyUI/view访问本地结果，支持重新打开工作流恢复预览。

## 来源与许可

结构来自BRIA VRMBG-3.0固定revision `59716e19a6cc97f91311edea190938219c097b76`，原model.py/config未改；不使用trust_remote_code。贡献者确认拥有此次BRIA源码公开集成/再分发授权，自己的封装和legacy_q授权MIT。BRIA源码仍为专有BRIA VRMBG-3.0许可，不属于根MIT；原模型卡及限制保留。随源未提供完整BRIA LICENSE文本，见THIRD_PARTY_NOTICES.md及上游模型卡；本仓库不授予模型使用或权重分发权。

源插件已记录多次大视频GPU、MOV/音频/Alpha16和生命周期验证；这些是源快照记录。此次RH集成验证以根README所列本次运行结果为准，源码原文件哈希见source_snapshot.json。
