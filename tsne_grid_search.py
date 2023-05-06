import tsne_original
import os 
import glob 
import torch 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler

def main():

    perplexities_low = [10, 20, 30, 50, 100, 150]
    pcas = [30, 50, 70, 100]
    experiences = ["Experience0", "Experience9"]

    model_path = os.path.join("model_outputs/*")
    list_of_files = glob.glob(model_path) # * means all if need specific format then *.csv
    model_output_filename = max(list_of_files, key=os.path.getctime)
    model_output = torch.load(model_output_filename)

    target_path = os.path.join("target_outputs/*")
    list_of_files = glob.glob(target_path) # * means all if need specific format then *.csv
    targets_filename = max(list_of_files, key=os.path.getctime)
    labels = torch.load(targets_filename).numpy()

    for experience in experiences:
        model_output[experience].requires_grad = False
        data = model_output[experience].numpy()
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
        data = StandardScaler().fit_transform(data)
        for perplexity in perplexities_low:
            for pca in pcas:
                if pca < perplexity:
                    continue
                else:
                    Y = tsne_original.tsne(X=data, no_dims=2, initial_dims=pca, perplexity=perplexity)
                    fig = plt.scatter(Y[:, 0], Y[:, 1], 10, labels)
                    fig.figure.savefig(f"./figures/exp={experience}_pca={pca}_perplexity={perplexity}")
                    plt.close(fig.figure)

if __name__ == "__main__":
    main()