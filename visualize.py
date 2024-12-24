import numpy as np
import matplotlib
import matplotlib.pyplot as plt    
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plotting
from train import prepare_dataset, get_embedding


def visualize_vectors_with_pca(data, labels=None, n_components=2):
    """
    Perform PCA on the given vector data and visualize it in 2D or 3D.

    Parameters:
        data (numpy.ndarray): The input vector data of shape (n_samples, n_features).
        labels (list or numpy.ndarray, optional): Labels for the data points. Default is None.
        n_components (int): Number of PCA components (2 or 3 for visualization).

    Returns:
        None: Displays the PCA plot.
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    # Create the plot
    fig = plt.figure()

    if n_components == 2:
        plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            s=50,
            alpha=0.7,
            edgecolors='k'
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization (2D)')
        # Add text labels
        if labels is not None:
            for i, label in enumerate(labels):
                plt.text(
                    reduced_data[i, 0],
                    reduced_data[i, 1],
                    str(label),
                    fontsize=9,
                    ha='right',
                    va='bottom'
                )
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            s=50,
            alpha=0.7,
            edgecolors='k'
        )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('PCA Visualization (3D)')
        # Add text labels
        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(
                    reduced_data[i, 0],
                    reduced_data[i, 1],
                    reduced_data[i, 2],
                    str(label),
                    fontsize=9
                )

    plt.show()


def visualize_vectors_with_tsne(data, names):
    embeddings = np.array(data)
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=50, alpha=0.7)

    # Annotate points with names
    for i, name in enumerate(names):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], name, fontsize=9, alpha=0.8)

    plt.title("t-SNE Visualization of Embeddings with Names")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    #data = np.random.rand(100, 5)    
    #labels = np.random.randint(0, 3, size=100)
    chunks, questions = prepare_dataset()
    assert len(chunks) == len(questions)
    data, labels, color_id = [], [], 0
    for i in range(6): #len(chunks)
        data.append( get_embedding(chunks[i]) )
        labels.append(f"{color_id}P")
        for question in questions[i]:
            data.append( get_embedding(question) )
            labels.append(f"{color_id}")
        color_id+=1

    #visualize_vectors_with_pca(data, labels=labels, n_components=2) #=2/3
    visualize_vectors_with_tsne(data, labels)
    
