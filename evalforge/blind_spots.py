import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def map_blind_spots(X, y_true, y_pred, n_clusters=5, random_state=42):
    """
    Finds hidden weak regions by clustering data and checking accuracy per cluster.

    If a cluster accuracy is below global accuracy, we mark it as a blind spot.
    Think of it as model eyesight testing, but without the eye chart :)

    Args:
        X (np.array): Feature matrix used for clustering.
        y_true (list or np.array): Ground truth labels.
        y_pred (list or np.array): Model predictions.
        n_clusters (int): Number of clusters for KMeans.
        random_state (int): Reproducibility seed.

    Returns:
        dict: Global accuracy, cluster-wise report, and blind spot summary.
    """
    X = np.array(X)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if X.shape[0] != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("X, y_true, and y_pred must all have matching sample counts.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot be greater than number of samples.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    global_accuracy = float(accuracy_score(y_true, y_pred))
    cluster_report = {}
    blind_spot_clusters = []

    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        cluster_size = int(np.sum(mask))

        if cluster_size == 0:
            cluster_accuracy = 0.0
        else:
            cluster_accuracy = float(accuracy_score(y_true[mask], y_pred[mask]))

        is_blind_spot = cluster_accuracy < global_accuracy
        if is_blind_spot:
            blind_spot_clusters.append(cluster_id)

        cluster_report[f"cluster_{cluster_id}"] = {
            "size": cluster_size,
            "accuracy": cluster_accuracy,
            "blind_spot": is_blind_spot,
        }

    return {
        "global_accuracy": global_accuracy,
        "n_clusters": n_clusters,
        "cluster_report": cluster_report,
        "blind_spot_clusters": blind_spot_clusters,
    }
