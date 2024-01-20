from BlogNN import BlogClassifier
from pathlib import Path

import torch
import os
import sys
import inspect
import json
import numpy as np

if __name__ == "__main__":
    """
        To run the script you need to provide 1 argument:
        name of JSON file with feature vector provided with
        'features' key.
    """
    cur_dir = Path(inspect.stack()[0][1]).parent
    weights_fp = os.path.join(cur_dir, '../models/blog_model')
    features_fp = os.path.join(cur_dir, '../JSONs/' + sys.argv[1])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_network = BlogClassifier().to(device)
    neural_network.load_state_dict(torch.load(weights_fp, map_location=torch.device('cpu')))
    neural_network.eval()  # Not necessary in our case but best practice

    with open(features_fp, 'r') as f:
        features = np.array(json.load(f)['features']).reshape((1, 770))
    features = torch.from_numpy(features).float()

    with torch.no_grad():
        prediction = neural_network(features.to(device))
        max_probability, matthew_probability = np.array(prediction.cpu()).reshape((2, 1)).tolist()
    max_probability = max_probability[0]
    matthew_probability = matthew_probability[0]

    print(f"Probabilities:")
    print(f"\t{max_probability * 100:.2f}% - Maxim's text")
    print(f"\t{matthew_probability * 100:.2f}% - Matthew's text")
    print()
    author = 'Maxim' if max_probability >= matthew_probability else 'Matthew'
    print(f"So I think that this text is written by: {author}")
