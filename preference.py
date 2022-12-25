from models import LSTM,ResNet,Bottleneck,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,GRU
def model_parameter(flag,hidden):
    if flag == 0:
        ##LSTM
        model_params = {
            'hiddenDim' : hidden,
            'bidirect' : True,
        }
    elif flag == 1:
        ##transformer
        model_params = {
            'heads' : 8,
            'depth' : 4,
        }
    elif flag == 2:
        ##cosformer
        model_params = {
            'use_cos': False,
            'kernel': 'elu',
            #'use_cos': True,
            #'kernel': 'relu',
            'd_model': 36,
            'n_heads': 4,
            'n_layers': 4,
            'ffn_ratio': 4,
            'rezero': False,
            'ln_eps': 1e-5,
            'denom_eps': 1e-5,
            'bias': False,
            'dropout': 0.2,    
            'xavier': True,
        }

    return model_params

def data_preference(cutoff,cutlen):
    dataset_size = 10000 
    
    cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : 10000,
        'stride' : 5000,
    }
    return dataset_size,cut_size

def model_preference(arch,hidden,classes,cutlen,learningrate,target,epoch):

    preference = {
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : epoch,
        "target" : target,
        "name" : arch
    }
    if "GRU" in str(arch):
        model_params = model_parameter(0,hidden)
        model = GRU(**model_params,**preference)
    elif "ResNet" in str(arch):
        model = ResNet(Bottleneck,[2,2,2,2],**preference)
    elif "Transformer" in str(arch):
        model_params = model_parameter(2,hidden)
        model = Transformer_clf_model(model_type='kernel', model_args=model_params,**preference)
    elif "LSTM" in str(arch):
        model_params = model_parameter(0,hidden)
        model = LSTM(**model_params,**preference)
    else:
        raise NotImplementedError("model selection error")
    useModel = arch
    return model,useModel