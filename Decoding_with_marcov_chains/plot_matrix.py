import io
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

with io.open('feynman_lectures_on_physics.txt', encoding='utf-8') as file:
    reference = file.readlines()

alphabet = set("abcdefghijklmnopqrstuvwxyz ")  # Use a set for faster membership testing


def letters_only(source):
    return ''.join(filter(lambda x: x in alphabet, source.lower()))


# Assuming 'reference' is defined somewhere
result = [letters_only(i).upper() for i in reference]

# Initialize transition matrix
trans_mat = np.zeros((27, 27), dtype=int)
letters = list(string.ascii_uppercase)
letters.append(' ')
index_dict = {letter: index for index, letter in enumerate(letters)}


# Function to update the matrix
def update_matrix(last_letter, current_letter):
    if last_letter:
        last_index = index_dict[last_letter]
        current_index = index_dict[current_letter]
        trans_mat[last_index, current_index] += 1


# Process the text
last_letter = ""
for line_number, line in enumerate(result):
    for char in line:
        if char in letters:
            update_matrix(last_letter, char)
            last_letter = char
        else:
            if last_letter:
                update_matrix(last_letter, '')
                last_letter = ""

x = []
for i in range(27):
    x.append(sum(trans_mat[i][j] for j in range(27)))

# Add one smoothing and calculate probabilities
trans_prob_mat = (trans_mat) / (trans_mat.sum(axis=1) + 1)[:, None]

# Convert to DataFrame for plotting
trans_prob_df = pd.DataFrame(trans_prob_mat, index=letters, columns=letters)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(trans_prob_df, cmap='Greys', linewidths=.5)
plt.title('Probability of Second Letter Given the First Letter')
plt.xlabel('Second Letter')
plt.ylabel('First Letter')
plt.show()
