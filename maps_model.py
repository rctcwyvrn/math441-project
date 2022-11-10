import time
import googlemaps
import numpy as np
import utils

api_key = open("googlemaps-apikey.txt", "r").read()
client = googlemaps.Client(key=api_key)

def make_set_covering_problem(positions):
    positions = [str(lat) + "," + str(long) for lat,long in positions]
    elems_per_min = 1000
    req_per_min = elems_per_min // len(positions)
    sleep_amt = int(60/req_per_min)
    print(f"Creating set covering matrix (n = {len(positions)}) (ratelimit sleep = {sleep_amt}s)")
    covering_matrix = np.array([])
    for start in positions:
        time.sleep(sleep_amt)
        # we can only have at most 25 destination, so we have to do this
        covered = []
        for slice_start_idx in range(0, len(positions), 25):
            # print(f"slice {slice_start_idx}, {slice_start_idx + 25}")
            # TODO: the covering matrix is symmetric, we can do half the number of elements requested from the api...
            destinations_slice = positions[slice_start_idx: slice_start_idx + 25]
            matrix = client.distance_matrix(origins=start, destinations=destinations_slice)
            times = [[elt["duration"]["value"] for elt in row["elements"]] for row in matrix["rows"]][0]
            covered.extend([1 if t < utils.NINE_MINS else 0 for t in times])
        assert(len(covered) == len(positions))
        covering_matrix = np.hstack((covering_matrix, covered))
    return covering_matrix.reshape(len(positions), len(positions))

    # the api restricts me to 100 elements per request, so this would fail on anything with more than 10 points
    # rip
    # matrix = client.distance_matrix(origins=positions, destinations=positions)
    # times = [[elt["duration"]["value"] for elt in row["elements"]] for row in matrix["rows"]]
    # covering_matrix = np.array([[1 if t < NINE_MINS else 0 for t in row] for row in times])
    # return covering_matrix.reshape(len(positions), len(positions))