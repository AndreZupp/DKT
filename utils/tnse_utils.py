import plotly.graph_objects as go
import plotly.express as px
import numpy as np 
from sklearn.preprocessing import StandardScaler
from src.tsne import tsne as original_tsne


def _single_tsne(model_output, n, pca, perplexity):
    model_output[f"Experience{n}"].requires_grad = False
    data = model_output[f"Experience{n}"].numpy()
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    data = StandardScaler().fit_transform(data)
    Y = original_tsne(data, no_dims =2, initial_dims = pca, perplexity = perplexity)
    return Y


def colors_from_labels(labels):
    color_dict = {x : px.colors.qualitative.Plotly[i] for i, x in enumerate(np.unique(labels))}
    color_dict = [color_dict[x] for x in labels]
    return color_dict

def plots_from_data(data, labels):
    plots = []
    color_dict = colors_from_labels(labels)
    for elem in data:
        plots.append(go.Scatter(x=elem[:, 0], y=elem[:,1], mode='markers', marker_color = color_dict, showlegend=False))
    return plots

def tsne(model_output, labels, n, pca, perplexity, target_exp=False, data_only=True):
    plots = []
    color_dict = colors_from_labels(labels)

    if n > 10 or n < 0:
        raise ValueError("n parameter must be 1 <= n <= 10")
    if n < 0 or n >= 10 and target_exp is True:
        raise ValueError("If target exp is True n must be 0 <= n <= 9")
    
    if not target_exp:
        for i in range(n):
            Y = _single_tsne(model_output, i, pca, perplexity)
            if not data_only:
                fig = go.Scatter(x = Y[:, 0], y = Y[:, 1], mode='markers', marker_color = color_dict)
                plots.append(fig)
            else:
                plots.append(Y)
            
    else:
        Y = _single_tsne(model_output, n, pca, perplexity)
        if not data_only:
                fig = go.Scatter(x = Y[:, 0], y = Y[:, 1], mode='markers', marker_color = color_dict)
                plots.append(fig)
        else:
            plots.append(Y)
        
    return plots if len(plots) > 1 else plots[0]
