import argparse
import os
import random

import numpy as np
from scipy import linalg
import pandas as pd
from tqdm.auto import tqdm

import image2vec_inception as ivi
import text2vec_cohere as tvc
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def cos_sim(vec_1, vec_2):
    assert vec_1.shape == vec_2.shape
    score = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return score
  
  
def matrix_sqrt(x):
    x = linalg.sqrtm(x, disp=False)[0].real
    return x


def feature_statistics(f):
    mu = np.mean(f, axis=0)
    sigma = np.cov(f, rowvar=False)
    return mu, sigma


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2, epsilon=1e-7):
    assert mu_1.shape == mu_2.shape
    assert sigma_1.shape == sigma_2.shape

    sse = np.sum(np.square(mu_1 - mu_2))
    covariance = matrix_sqrt(sigma_1 @ sigma_2)
        
    if np.isinf(covariance).any():
        I = np.eye(sigma_1.shape[0])
        covariance = matrix_sqrt(sigma_1 @ sigma_2 + epsilon * I)

    fid = sse + np.trace(sigma_1) + np.trace(sigma_2) - 2 * np.trace(covariance)
    return fid


def fid(f_1, f_2):
    # It is recommended, however, to compute feature_statistics only once and store the resulting
    # mu and sigma instead of recalculating it every time with this function
    assert f_1.shape == f_2.shape
    mu_1, sigma_1 = feature_statistics(f_1)
    mu_2, sigma_2 = feature_statistics(f_2)
    fid = frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
    return fid


def calc_user_similarity(user_1, user_2):
    '''
    Finds the normalized similarity score between 2 users given their vectors as lists
    normalized_similarity ranges from [-1, 1]
    '''
    similarity_scores = []

    for vec1 in user_1:
        for vec2 in user_2:
            similarity_scores.append(cos_sim(vec1, vec2))

    sum_similarity = sum(similarity_scores)
    num_pairs = len(similarity_scores)

    normalized_similarity = sum_similarity / num_pairs
    return normalized_similarity

if __name__ == "__main__":
    MAX_VIDEOS = 750
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_top", default=MAX_VIDEOS, type=int)
    parser.add_argument("-t", "--type", choices=["title", "thumbnail"],
                        default="title", type=str)
    parser.add_argument("-d", "--dir", default="./youtube/watch-histories", type=str)
    args = parser.parse_args()
    
    histories_file = sorted(os.listdir(args.dir))
    data = []
    for file in histories_file:
        history = pd.read_csv(f"{args.dir}/{file}", sep=",")[:args.num_top]
        #print(history['title'])
        data_ = history[args.type].tolist()
        data.append(data_)
    
    features = []
    stats = []
    for data_ in tqdm(data):
        if args.type == "title":
            features_ = tvc.text2vec(data_)
        elif args.type == "thumbnail":
            features_ = ivi.link2vec(data_, N=args.num_top // 10)
        features.append(features_)
        mu, sigma = feature_statistics(features_)
        stats.append((mu, sigma))
    
    for i in range(len(stats)):
        name = histories_file[i][:histories_file[i].find('.')]
        np.save(f"./youtube/features.npy", features[i])
        np.save(f"./youtube{args.type}_{name}_stats_mu.npy", stats[i][0])
        np.save(f"./youtube{args.type}_{name}_stats_sigma.npy", stats[i][1])
        for j in range(i + 1, len(stats)):
            # compute similarities using different metrics between all possible pairs of users based on YouTube history
            mu_1, sigma_1 = stats[i]
            mu_2, sigma_2 = stats[j]
            fid = frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
            print(f"{histories_file[i]}\t{histories_file[j]}\t{fid}")
           
            cos_sim_val = np.mean(cos_sim(features[i], features[j]))
            euclidean_dist_val = np.mean(euclidean_distances(features[i], features[j]))

            comparison_results = comparison_results.append({
                "User_1": "Aaron",
                "User_2": "Friend",
                "Frechet Distance": fid,
                "Cosine Similarity": cos_sim_val,
                "Euclidean Distance": euclidean_dist_val
            }, ignore_index=True)

    comparison_results.to_csv("user_similarities.csv", index=False)
