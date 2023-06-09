import torch
"""
属 genus、 科 family、目 order、綱 class、門 phylum、界 kingdom、超界 domain
"""
def base_data(data):
      return torch.cat(data)

def base_labels(data,label):
      label_list = torch.zeros(1)
      for idx,l in enumerate(label):
            label_list = torch.hstack((label_list,torch.ones(data[idx].shape[0])*l))
      return label_list[1:].to(torch.int64)

def classification(mode,length):
      if mode == "order":
            # return [0,1,2,3,3,4]
            return [0,1,2,3,2,2,2,2,3,1,0,2]
      elif mode == "family":
            # return [0,1,1,1,1,2]
            return [0,1,2,3,2,2,2,4,5,6,7,8]
      else:
            return [i for i in range(length)]


class MultiDataset(torch.utils.data.Dataset):
      captions = None
      classes = 0
      # @classmethod
      # def caption():
      #       return MultiDataset.captions
      # @classmethod
      # def num_class():
      #       return MultiDataset.classes

      def __init__(self, data:list,mode:str):
            label = classification(mode,len(data))
            if len(label) != len(data):
                  raise IndexError("label size does not match to dataset size")
             
            self.data = base_data(data)
            self.label = base_labels(data,label)

            self.__class__.captions = label
            self.__class__.classes = max(label)+1

      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y


