from transformers import PretrainedConfig


class VRMBG3Config(PretrainedConfig):
    model_type = "vrmbg3"

    def __init__(self, in_channels=6, **kwargs):
        self.in_channels = in_channels
        super().__init__(**kwargs)
