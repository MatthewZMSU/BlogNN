from .BlogNN import BlogClassifier
from pathlib import Path

import torch
import os
import sys
import inspect
import json
import numpy as np


def model_predict(feature_list):
    """
        To run the script you need to provide 1 argument:
        name of JSON file with feature vector provided with
        'features' key.
    """
    cur_dir = Path(inspect.stack()[0][1]).parent
    weights_fp = os.path.join(cur_dir, '../models/blog_model')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_network = BlogClassifier().to(device)
    neural_network.load_state_dict(torch.load(weights_fp, map_location=torch.device('cpu')))
    neural_network.eval()  # Not necessary in our case but best practice

    features = np.array(feature_list).reshape((1, 770))
    features = torch.from_numpy(features).float()

    with torch.no_grad():
        prediction = neural_network(features.to(device))
        max_probability, matthew_probability = np.array(prediction.cpu()).reshape((2, 1)).tolist()
    max_probability = max_probability[0]
    matthew_probability = matthew_probability[0]

    return max_probability, matthew_probability
