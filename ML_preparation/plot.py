import matplotlib.pyplot as plt

def plot_torch_1d(data,idx)-> None:
    print("##### DATA PLOT #####")
    plt.plot(data.cpu().numpy())
    plt.savefig(f'test_plot_{idx}.png') 
    plt.clf()