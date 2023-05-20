import gc
import torch
from pytorch_lightning.callbacks import Callback

def free_memory():
    """(Maybe) prevents Cuda running out of memory
    """
    gc.collect()
    torch.cuda.empty_cache()

class Garbage_collector_callback(Callback):
    """Custom callback that (maybe) prevents Cuda running out of memory.
    I have absolutely no idea if this actually helps. However, Cuda on Colab
    is prone to memory leaks, especially in case of Ctrl + C interrupts. 
    After using this callback the issue kinda disappeared. Code based on 
    https://huggingface.co/transformers/main_classes/callback.html
    """
    def on_train_end(self, trainer, pl_module):
        """Called every time the Trainer logs data.
        """
        res_before = torch.cuda.memory_reserved(0)
        free_memory()
        res_after = torch.cuda.memory_reserved(0)
        freed = res_before - res_after
        print(f'Freed {freed}.')