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
            #'use_cos': False,
            #'kernel': 'elu',
            'use_cos': True,
            'kernel': 'relu',
            'd_model': 112,
            'n_heads': 8,
            'n_layers': 3,
            'ffn_ratio': 8,
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

def model_preference(arch,hidden,classes,cutlen,learningrate,target,epoch,heatmap):
    cnn_params = {
        "out_dim" : 96,
        "kernel" : 14,
        "stride" : 2,
    }
    #{'out_dim': 91, 'kernel': 14, 'stride': 2}
    #out_dim': 112.0, 'kernel': 17, 'stride': 5, 'n_layers': 3, 'ffn_ratio': 8

    preference = {
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : epoch,
        "target" : target,
        "name" : arch,
        "heatmap" : heatmap,
    }
    if "GRU" in str(arch):
        model_params = model_parameter(0,hidden)
        model = GRU(cnn_params,preference,**model_params)
    elif "ResNet" in str(arch):
        model = ResNet(Bottleneck,[2,2,2,2],preference)
    elif "Transformer" in str(arch):
        model_params = model_parameter(2,hidden)
        model = Transformer_clf_model(cnn_params,model_type='kernel', model_args=model_params,**preference)
    elif "LSTM" in str(arch):
        model_params = model_parameter(0,hidden)
        model = LSTM(**model_params,**preference)
    else:
        raise NotImplementedError("model selection error")
    useModel = arch
    return model,useModel