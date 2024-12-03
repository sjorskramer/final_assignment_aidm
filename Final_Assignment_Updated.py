import numpy as np
import scipy as sp
import sys
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
import time
from joblib import Parallel, delayed


threshold = 0.5


def import_data():
    """
    Import the data and convert it to a sparse matrix
    Hint 1: Think about the right way to represent the data. I.e. use the sparse package from the Scipy library.
    """
    # Load the user_movie_rating file
    df = pd.DataFrame(np.load('user_movie_rating.npy'), columns=['user_id', 'movie_id', 'rating'])

    # Convert the DataFrame to a sparse matrix
    rows = df['user_id'].values
    cols = df['movie_id'].values
    data = df['rating'].values
    sparse_matrix = csr_matrix((data, (rows, cols)))  # research found CSR has more efficient row slicing

    return sparse_matrix


def minhashing(sparse_matrix):
    """
    Function for the minhashing
    Hint 2: Use random permutations of columns or rows, instead of hash functions. This avoids time-consuming loops
    Hint 3: Relatively short signatures (80-150) should result in sufficiently good results (and take less time to compute).
    """

    n_perms = 100  # length of the signatures
    n_users = sparse_matrix.shape[0]
    signatures = np.zeros((n_users, n_perms), dtype=int)

    for i in range(n_perms):
        perm = np.random.permutation(sparse_matrix.shape[1])  # permuting the columns/movies
        for user_index in range(n_users):
            for col_index in sparse_matrix[user_index].nonzero()[1]:  # finding the first rated movie
                signatures[user_index, i] = perm[col_index]
                break
    return signatures


def jaccard_similarity(u1, u2, sparse_matrix):
    """
    Function to compute Jaccard similarity between two users based on their ratings.
    """
    # Find the rated movies for both users
    u1_ratings = set(sparse_matrix[u1].indices)
    u2_ratings = set(sparse_matrix[u2].indices)

    # Compute the Jaccard similarity
    intersection = len(u1_ratings & u2_ratings)
    union = len(u1_ratings | u2_ratings)

    if union == 0:
        return 0
    return intersection / union


def local_sensitive_hashing(signatures, sparse_matrix):
    """
    Function for the LSH algorithm
    Hint 4: You have to check k(k-1)/2 pairs, when the bucket has k elements. Postpone evaluation of such a bucket
    until the very end (or just ignore it – they are really expensive). Or better: consider increasing the number
    of rows per band – that will reduce the chance of encountering big buckets.
    Hint 5: Note that b*r doesn’t have to be exactly the length of the signature. For example, when you work with
    signatures of length n = 100, you may consider e.g., b = 3, r = 33; b = 6, r = 15, etc.
    """
    num_bands = 3
    rows_per_band = 33
    num_users, num_perms = signatures.shape

    # Parallelize band processing
    all_candidates = Parallel(n_jobs=2)(
        delayed(process_band)(band, num_bands, rows_per_band, signatures, num_users)
        for band in range(num_bands)
    )

    # Combine all candidate pairs after parallel processing
    candidate_pairs = set()
    for candidates in all_candidates:
        candidate_pairs.update(candidates)

    # Calculate Jaccard similarity for candidate pairs
    similar_pairs = []
    for u1, u2 in candidate_pairs:
        similarity = jaccard_similarity(u1, u2, sparse_matrix)
        if similarity >= threshold:
            similar_pairs.append((u1, u2, similarity))

    return similar_pairs


def process_band(band, num_bands, rows_per_band, signatures, num_users):
    """
    Helper function to process each band and generate candidate pairs
    """
    start_row = band * rows_per_band
    end_row = start_row + rows_per_band
    band_buckets = defaultdict(list)

    for user_index, signature in enumerate(signatures):
        band_signature = tuple(signature[start_row:end_row])
        band_buckets[band_signature].append(user_index)

    candidate_pairs = set()
    for users in band_buckets.values():
        if len(users) > 1:
            if len(users) > 1000:  # Skip large buckets
                continue
            candidate_pairs.update((users[i], users[j]) for i in range(len(users)) for j in range(i + 1, len(users)))

    return candidate_pairs

def add_to_result(similar_pairs, filename):
    """
    Add the similarity above the threshold to the result.txt
    Hint 6: Open and close the file after writing to the file, to save time.
    """

    with open(filename, "w", buffering=8192) as f:
        lines = [f"{min(u1, u2)},{max(u1, u2)}\n" for u1, u2, _ in similar_pairs]
        f.writelines(lines)
        f.write(f"\nNumber of similar pairs: {len(similar_pairs)}\n")


if __name__ == "__main__":
    randomseed = sys.argv[1] if len(sys.argv) > 1 else "seed"  # Hint 8: Do not cherry-pick a good seed.
    start_time = time.time()
    filename = "result.txt"

    # Import and load data
    data = import_data()
    print("First function completed")
    signatures = minhashing(data)
    print("Second function completed")


    # Run LSH and calculate similar pairs
    similar_pairs = local_sensitive_hashing(signatures, data)
    print("Third function completed")

    # Save results to file
    add_to_result(similar_pairs, filename)
    print("Fourth function completed")

    # End timer and calculate runtime of the algorithm in minutes
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Results saved to {filename}. Elapsed time: {elapsed_time:.2f} minutes.")




