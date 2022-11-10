import numpy as np
from geopy.distance import geodesic

def make_set_covering_problem_straight_line(positions):
    positions = [str(lat) + "," + str(long) for lat,long in positions]
    print(f"Creating set covering matrix (n = {len(positions)}) (straight line distances)")
    covering_matrix = np.array([])
    for start in positions:
        # we can only have at most 25 destination, so we have to do this
        covered = []
        for idx, other in enumerate(positions):
            if geodesic(start, other).kilometers < 3.0:
                covered.append(1)
            else:
                covered.append(0)
        covering_matrix = np.hstack((covering_matrix, covered))
    return covering_matrix.reshape(len(positions), len(positions))