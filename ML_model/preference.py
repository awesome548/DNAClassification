from ML_model import LSTM, resnet, effnetv2_s, gru, cosformer

DEFAULT_CNN = {
    "channel": 20,
    "kernel": 19,
    "stride": 3,
    "padd": 5,
}

EFFNETV2_CFGS = [
    # t, c, n, s, SE
    [1,  24,  2, 1, 0],
    [4,  48,  4, 2, 0],
    [4,  64,  4, 2, 0],
    [4, 128,  6, 2, 1],
    [6, 160,  6, 1, 1],
    [6, 256,  6, 2, 1],
]

_RNN_FLAG = 0
_TRANSFORMER_FLAG = 1
_COSFORMER_FLAG = 2


def model_parameter(flag: int, hidden: int) -> dict:
    if flag == _RNN_FLAG:
        return {
            "hiddenDim": hidden,
            "bidirect": True,
        }
    elif flag == _TRANSFORMER_FLAG:
        return {
            "heads": 8,
            "depth": 4,
        }
    elif flag == _COSFORMER_FLAG:
        return {
            "use_cos": False,
            "kernel": "elu",
            "d_model": 112,
            "n_heads": 8,
            "n_layers": 3,
            "ffn_ratio": 8,
            "rezero": False,
            "ln_eps": 1e-5,
            "denom_eps": 1e-5,
            "bias": False,
            "dropout": 0.2,
            "xavier": True,
        }
    raise ValueError(f"Unknown model flag: {flag}")


def model_preference(arch: str, hidden: int, pref: dict, mode: int = 0) -> tuple:
    if "GRU" in str(arch):
        params = model_parameter(_RNN_FLAG, hidden)
        model = gru(param=params, preference=pref)
    elif "ResNet" in str(arch):
        model = resnet(mode=mode, preference=pref)
    elif "Transformer" in str(arch):
        params = model_parameter(_COSFORMER_FLAG, hidden)
        model = cosformer(preference=pref, args=params)
    elif "LSTM" in str(arch):
        params = model_parameter(_RNN_FLAG, hidden)
        model = LSTM(**params, **pref)
    elif "Effnet" in str(arch):
        model = effnetv2_s(mode=mode, preference=pref)
    else:
        raise NotImplementedError(f"Unknown architecture: {arch}")
    return model, arch
