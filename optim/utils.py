from ray import tune
import optuna.distributions as op

def resnet_param():
    search_space = {
        'cutlen': tune.qrandint(1000,9000,500),
        'conv_1': tune.qrandint(16,128,16),
        'conv_2': tune.qrandint(16,128,16),
        'conv_3': tune.qrandint(16,128,16),
        'conv_4': tune.qrandint(16,128,16),
        'layer_1': tune.randint(1,5),
        'layer_2': tune.randint(1,5),
        'layer_3': tune.randint(1,5),
        'layer_4': tune.randint(1,5),
        'lr' : tune.qloguniform(1e-3, 5e-1, 5e-4), 
    }
    return search_space

def effnet_param(trial):
    cutlen = int(trial.suggest_float("cutlen",1000,9000,step=1000))
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

def effnet_var(config):
    cutlen = config['cutlen']
    channel = config['channel']
    kernel = config['kernel']
    stride = config['stride']
    padd = config['padd']
    mode = config['mode']
    lr = config['lr']
    cnn_params = {
        "channel" : channel,
        "kernel" : kernel,
        "stride" : stride,
        "padd" : padd,
    }
    return cutlen,cnn_params,mode,lr
def resnet_var(config):
    cutlen = config['cutlen']
    conv_1 = config['conv_1']
    conv_2 = config['conv_2']
    conv_3 = config['conv_3']
    conv_4 = config['conv_4']
    layer_1 = config['layer_1']
    layer_2 = config['layer_2']
    layer_3 = config['layer_3']
    layer_4 = config['layer_4']
    learningrate = config['lr']
    cfgs = [
        [conv_1,layer_1],
        [conv_2,layer_2],
        [conv_3,layer_3],
        [conv_4,layer_4],
    ]
    return cutlen,conv_1,conv_2,conv_3,conv_4,layer_1,layer_2,layer_3,layer_4,learningrate,cfgs