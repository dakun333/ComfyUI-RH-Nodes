"""CF1 with the fixed ST1 narrow transition and PO1 outlier retreat."""
import json
from comfy_api.latest import io
from .guarded_nodes import _Y367GuardedDestripe, ACTUAL8, FLOAT32, ORIENTATIONS


class _Y367ProtectedDestripe(io.ComfyNode, _Y367GuardedDestripe):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID, display_name=cls.METHOD_NAME,
            category=_Y367GuardedDestripe.CATEGORY,
            description="保留原颜色与轮廓保护，增加ST1窄过渡及PO1异常删除量回撤。只回撤已允许的RGB校正，不扩散到原零区。" + _Y367GuardedDestripe.DESCRIPTION,
            search_aliases=[cls.VARIANT.split("_")[0], "ST1", "PO1", "异常保护", "横纹", "竖纹"],
            inputs=[
                io.Image.Input("image", tooltip="原生8位RGB/RGBA图片；保持原尺寸和透明度。"),
                io.Combo.Input("orientation", options=list(ORIENTATIONS), default="both", tooltip="vertical竖纹，horizontal横纹，both联合；固定48周期。"),
                io.Combo.Input("output_mode", options=[ACTUAL8, FLOAT32], default=ACTUAL8, tooltip="推荐actual8接标准Save Image；float32返回保护后的FLOAT。"),
                io.Mask.Input("mask", optional=True, tooltip="Load Image透明度MASK，1透明、0不透明；不是去纹范围。"),
            ],
            outputs=[
                io.Image.Output(display_name="处理图像", tooltip=_Y367GuardedDestripe.OUTPUT_TOOLTIPS[0]),
                io.Image.Output(display_name="删除量×16", tooltip=_Y367GuardedDestripe.OUTPUT_TOOLTIPS[1]),
                io.String.Output(display_name="处理报告", tooltip="包含ST1/PO1回撤量、实际推理次数和处理时间。"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _Y367GuardedDestripe.IS_CHANGED.__func__(cls, **kwargs)

    @classmethod
    def execute(cls, image, orientation="both", output_mode=ACTUAL8, mask=None):
        output, diagnostic, _, text = cls().process(image, orientation, output_mode, mask)
        report = json.loads(text)
        report["full_logical_model_inferences_per_image"] = report["model_inferences_per_image"]
        report["model_inferences_per_image"] = [frame["model_inferences_executed"] for frame in report["frames"]]
        report["protection_policy"] = "ST1 narrow transition + PO1"
        return io.NodeOutput(output, diagnostic, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


class Y367CF1ProtectedDestripe(_Y367ProtectedDestripe):
    VARIANT = "CF1_PO1"
    NODE_ID = "Y367CF1ProtectedDestripe"
    METHOD_NAME = "Y367-FAST CF1 异常保护"


NODE_CLASS_MAPPINGS = {Y367CF1ProtectedDestripe.NODE_ID: Y367CF1ProtectedDestripe}
NODE_DISPLAY_NAME_MAPPINGS = {cls.NODE_ID: cls.METHOD_NAME for cls in NODE_CLASS_MAPPINGS.values()}
