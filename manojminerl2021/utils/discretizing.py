import joblib
import numpy as np
import logging
from sklearn.cluster import KMeans

HDBSCAN_PARAMS = {'min_samples': 10, 'prediction_data': True}

class Discretizer:
    def __init__(self, n_actions=None):
        self.n_actions = n_actions
        self.clusterer = None
        self.means = []
        self.labels = None

    def cluster_action_data(self, data: np.ndarray):
        if self.n_actions is None:
            raise ValueError('`n_actions` attribute of discretizer not set')
        logging.info('running clustering on action data with {} samples'.format(data.shape[0]))
        self.clusterer = KMeans(n_clusters=self.n_actions, n_jobs=-1, max_iter=1000).fit(data)
        self.means = self.clusterer.cluster_centers_
    
    def cluster_means(self, data, assignments):
        labels = np.unique(assignments >= 0)
        clusters = [[] for _ in range(len(labels))]
        for i, label in assignments:
            if label >= 0:
                clusters[label].append(data[i])
        
        clusters = [np.array(c) for c in clusters]
        self.means = [c.mean(axis=0) for c in clusters]

    def make_continuous(self, action: int):
        assert action < self.n_actions and action >= 0, 'invalid action index, must be between 0 and {}'.format(self.n_actions)
        return self.means[action]

    def make_discrete(self, action: np.ndarray):
        raise NotImplementedError

    def load_cluster_data(self, load_file):
        with open(load_file, 'rb') as f:
            self.labels, self.n_actions, self.means = joblib.load(f)
        
        logging.info('loaded clustering data with {} samples and {} actions'.format(len(self.labels), self.n_actions))