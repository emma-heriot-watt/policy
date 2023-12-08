import numpy as np


def levenshtein_distance(string1: str, string2: str) -> int:
    """Levenheisten distance powered by ChatGPT.

    The Levenshtein distance is a string metric for measuring the difference between two sequences.
    Informally, the Levenshtein distance between two words is the minimum number of single-
    character edits (insertions, deletions or substitutions) required to change one word into the
    other.
    """
    # Create a matrix of distances between all prefixes of the two strings
    m = len(string1)
    n = len(string2)
    similarity_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    similarity_matrix[:, 0] = np.arange(0, m + 1)
    similarity_matrix[0, :] = np.arange(0, n + 1)

    # Fill in the rest of the matrix
    for idx in range(1, m + 1):
        for jdx in range(1, n + 1):
            if string1[idx - 1] == string2[jdx - 1]:
                # If the last characters are the same, the distance is the same as
                # the distance between the strings without those characters
                cost = 0
            else:
                # If the last characters are different, we need to perform a
                # substitution which has a cost of 1
                cost = 1

            similarity_matrix[idx][jdx] = min(
                similarity_matrix[idx - 1][jdx] + 1,  # Deletion
                similarity_matrix[idx][jdx - 1] + 1,  # Insertion
                similarity_matrix[idx - 1][jdx - 1] + cost,  # Substitution
            )

    # The distance is the distance between the full-length strings
    return similarity_matrix[m][n]
