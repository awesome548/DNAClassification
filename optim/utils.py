from ray import tune
import optuna.distributions as op

def resnet_param(trial):
    cutlen = 3000
    channel = trial.suggest_int("channel",20,128)
    c1 = trial.suggest_int("c1",20,128)
    c2 = trial.suggest_int("c2",20,128)
    c3 = trial.suggest_int("c3",20,128)
    c4 = trial.suggest_int("c4",20,128)
    kernel = trial.suggest_int("kernel",15,30)
    stride = trial.suggest_int("stride",1,5)
    padd = trial.suggest_int("padd",1,5)
    mode = trial.suggest_int("mode",0,1),
    lr = trial.suggest_float("lr",1e-3, 5e-1,log=True)
    cfgs = [
        [c1,2],
        [c2,2],
        [c3,2],
        [c4,2]
    ]
    cnn_params = {
        "channel" : channel,
        "kernel" : kernel,
        "stride" : stride,
        "padd" : padd,
    }
    return cutlen,mode,lr,cnn_params,cfgs

def effnet_param(trial):
    cutlen = int(trial.suggest_float("cutlen",5000,9000,step=1000))
    channel = trial.suggest_int("channel",20,128)
    kernel = trial.suggest_int("kernel",15,30)
    stride = trial.suggest_int("stride",1,5)
    padd = trial.suggest_int("padd",1,5)
    mode = trial.suggest_int("mode",0,1),
    lr = trial.suggest_float("lr",1e-3, 5e-1,log=True)
    cnn_params = {
        "channel" : channel,
        "kernel" : kernel,
        "stride" : stride,
        "padd" : padd,
    }
    return cutlen,mode,lr,cnn_params
