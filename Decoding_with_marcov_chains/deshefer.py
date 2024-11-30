import random
from string import ascii_uppercase

import numpy as np

from Decoding_with_marcov_chains.plot_matrix import letters, trans_prob_mat


def decode(mapping, coded):
    coded = coded.upper()
    decoded = list(coded)
    for i, char in enumerate(coded):
        if char in letters:
            decoded[i] = letters[mapping.index(char)]
    return ''.join(decoded)


def log_prob(mapping, decoded, trans_prob_mat):
    logprob = 0
    lastletter = ""
    for char in decoded:
        curletter = char
        if curletter in ascii_uppercase:
            logprob += np.log(trans_prob_mat[ascii_uppercase.index(lastletter), ascii_uppercase.index(curletter)])
            lastletter = curletter
        else:
            if lastletter != "":
                logprob += np.log(trans_prob_mat[ascii_uppercase.index(lastletter), 26])
                lastletter = ""
    if lastletter != "":
        logprob += np.log(trans_prob_mat[ascii_uppercase.index(lastletter), 26])
    return logprob


# Assuming trans_prob_mat is defined elsewhere in your Python code
# trans_prob_mat = np.zeros((27, 27)) # Placeholder for the transition probability matrix
correctTxt = """
"in the study of physics usually the course is divided
into a series of subjects such as mechanics electricity
optics etc and one studies one subject after the other
for example this course has so far dealt mostly with mechanics
but a strange thing occurs again and again the equations
which appear in different fields of physics and even in
other sciences, are often almost exactly the same so that
many phenomena have analogs in these different fields"
"""
correctTxt = correctTxt.upper()

# if you want to start with no random change than replace text 3 code lines for this 3:
# coded = correctTxt
# mapping = letters
# i = 0

coded = decode(random.sample(letters, len(letters)), correctTxt)
mapping = random.sample(letters, len(letters))
i = 0

iters = 200000
cur_decode = decode(mapping, coded)
cur_loglike = log_prob(mapping, cur_decode, trans_prob_mat)
max_loglike = cur_loglike
max_decode = cur_decode

while i <= iters:
    proposal = random.sample(range(27), 2)
    prop_mapping = mapping.copy()
    prop_mapping[proposal[0]], prop_mapping[proposal[1]] = mapping[proposal[1]], mapping[proposal[0]]
    prop_decode = decode(prop_mapping, coded)
    prop_loglike = log_prob(prop_mapping, prop_decode, trans_prob_mat)
    if random.uniform(0, 1) < np.exp(prop_loglike - cur_loglike):
        mapping = prop_mapping
        cur_decode = prop_decode
        cur_loglike = prop_loglike
        if cur_loglike > max_loglike:
            max_loglike = cur_loglike
            max_decode = cur_decode

        print(i, cur_decode, cur_loglike)
        i += 1
