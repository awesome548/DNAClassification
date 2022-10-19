import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):
        super(Predictor,self).__init__
        
        self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.label = nn.Linear(hiddenDim, outputDim)

    
    #初期状態の引数を渡す
    def forward(self,inputs,hidden0=None):
        # h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # return self.label(final_hidden_state[-1]) 
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        return self.label(output[:,-1,:])
        
