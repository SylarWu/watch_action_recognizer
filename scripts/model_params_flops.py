import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table


def get_backbone(model_name: str):
    from model import resnet, ResNetConfig
    from model import vit, TransformerConfig
    from model import mlp_mixer, MLPMixerConfig
    from model import lstm, LSTMConfig
    if model_name.startswith('resnet'):
        return resnet(model_name, ResNetConfig())
    elif model_name.startswith('vit'):
        return vit(model_name, TransformerConfig())
    elif model_name.startswith('mixer'):
        return mlp_mixer(model_name, MLPMixerConfig())
    elif model_name.startswith('lstm'):
        return lstm(model_name, LSTMConfig())


def get_params_flops(model_name: str):
    backbone = get_backbone(model_name)
    accData, gyrData = torch.randn(1, 3, 224), torch.randn(1, 3, 224)
    flops = FlopCountAnalysis(backbone, (accData, gyrData))
    print(model_name.center(100, "="))
    print(flop_count_table(flops))


if __name__ == '__main__':
    model_list = [
        "vit_es_8", "vit_es_16", "vit_es_32",
        "vit_ms_8", "vit_ms_16", "vit_ms_32",
        "vit_s_8", "vit_s_16", "vit_s_32",
        "mixer_es_8", "mixer_es_16", "mixer_es_32",
        "mixer_ms_8", "mixer_ms_16", "mixer_ms_32",
        "mixer_s_8", "mixer_s_16", "mixer_s_32",
        "lstm_es", "lstm_ms", "lstm_s",
        "resnet18", "resnet34", "resnet50", "resnet101",
    ]

    for model_name in model_list:
        get_params_flops(model_name)
