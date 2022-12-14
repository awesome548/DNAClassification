from models import CNNLstmEncoder,ResNet,Bottleneck,SimpleViT,ViT,ViT2,SimpleViT2
def model_parameter(classes,hidden):
    cnn_params = {
        'padd' : 5,
        'ker' : 19,
        'stride' : 3,
        'convDim' : 16,
    }
    lstm_params = {
        'inputDim' : 1,
        'hiddenDim' : hidden,
        'outputDim' : classes,
        'bidirect' : True,
    }
    transformer_params = {
        'classes' : classes,
        'heads' : 8,
        'depth' : 4,
    }
    return cnn_params,lstm_params,transformer_params

def data_preference(transform,cutoff,cutlen):
    base_classes = 6
    dataset_size = 12000 
    inputDim = 1

    data_transform = {
        'isFormat' : transform,
        'dim' : inputDim,
        'length' : cutlen,
    }
    cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : 10000,
        'stride' : 5000,
    }
    return base_classes,dataset_size,data_transform,cut_size

def model_preference(arch,lstm_params,transformer_params,classes,cutlen,learningrate):
    useResNet = False
    useTransformer = False
    useLstm = False
    if "ResNet" in str(arch):
        transform = False
        useResNet = True
    elif "LSTM" in str(arch):
        transform = True
        useLstm = True
    else:
        assert str(arch) == "Transformer"
        transform = True
        useTransformer = True
    useModel = arch
    if useLstm:
        model = CNNLstmEncoder(**lstm_params,lr=learningrate,classes=classes)
    elif useResNet:
        model = ResNet(Bottleneck,[2,2,2,2],classes=classes,cutlen=cutlen,lr=learningrate)
    elif useTransformer:
        # model = ViT2(**transformer_params,length=cutlen,lr=learningrate)
        model = SimpleViT2(**transformer_params,lr=learningrate)
    return model,transform,useModel