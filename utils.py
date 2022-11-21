import googlemaps
import requests
from IPython.display import display
import math
import numpy as np
from scipy.optimize import linprog
from PIL import Image
import io
import cv2
import string
import time
from global_land_mask import globe
from geopy.distance import geodesic, distance
import random

api_key = open("googlemaps-apikey.txt", "r").read()
client = googlemaps.Client(key=api_key)
# client = None # temporarily disabled

# def get_map(center, markers=[], marker_color="blue", zoom=13, fill=False):
def get_map(center, markers=[], fills=[], zoom=13):
    static_map_url = "https://maps.googleapis.com/maps/api/staticmap?"
    lat,long = center
    center = str(lat)+","+str(long)
    req = f"{static_map_url}center={center}&zoom={str(zoom)}&size=400x400&key={api_key}&sensor=false"
    
    for color, points in markers:
        req += f"&markers=color:{color}|" + "|".join([str(lat)+","+str(long) for lat,long in points])

    for points in fills:
        path = "&path=color:0x00000000|weight:5|fillcolor:0xf55742|" + "|".join([str(lat)+","+str(long) for lat,long in points])
        req = req + path

    r = requests.get(req)
    if "X-Staticmap-API-Warning" in r.headers:
        print(r.headers["X-Staticmap-API-Warning"])
        return None
    return Image.open(io.BytesIO(r.content))

# parsing the distance matrix
# times = client.distance_matrix(origins=center, destinations=circle)
# print("\n".join([str([elt["duration"]["text"] for elt in row["elements"]]) for row in times["rows"]]))

NINE_MINS=9*60
COVERABLE_LOWER_BOUND=8.75*60
COVERABLE_UPPER_BOUND=9.25*60

# lat,long -> encoded polyline representing the region
def distance_coverable(center):
    print("DISABLED")
    return 

    n = 25
    r = 0.02
    # def circle_offsets():
    #     offsets =[]
    #     for i in range(n):
    #         deg = i/n * 2 * math.pi
    #         offsets.append([math.cos(deg)*r, math.sin(deg)*r])
    #     return offsets

    # circle = [[i, center[0] + offset[0], center[1] + offset[1]] for i, offset in enumerate(circle_offsets())]
    circle = []
    for i in range(n):
        rad = i/n * 2 * math.pi
        deg = 180 * rad/math.pi 
        point = distance(r).destination(center, bearing=deg)
        circle.append([i, point.latitude, point.longitude])

    unchanged = False
    max_iterations = 5
    num_iter = 0
    while num_iter < max_iterations and not unchanged:
        # display(get_map(center, markers=[("red",[(lat,long) for (_, lat, long) in circle])], fills=[[(lat,long) for (_, lat, long) in circle]], zoom=11))
        for point in list(circle):
            if globe.is_ocean(point[1], point[2]):
                circle.remove(point)
        num_iter += 1
        unchanged = True
        times = client.distance_matrix(origins=center, destinations=[(lat,long) for (i, lat, long) in circle])
        # print(times)
        matrix = [[elt["duration"]["value"] for elt in row["elements"]] for row in times["rows"]]
        # note: sometimes elt['status'] might be NO_RESULTS, and this line will error out
        pairs = zip(circle, matrix[0])
        circle = []
        for ((i, lat, long), time) in pairs:
            deg = i/n * 2 * math.pi
            y = (lat - center[0])
            x = (long - center[1])
            old_r = math.sqrt(x ** 2 + y ** 2)
            # print(f"deg = {deg}, old_r = {old_r}, time {time}")
            if time < COVERABLE_LOWER_BOUND or time > COVERABLE_UPPER_BOUND:
                if time == 0:
                    ratio = 10
                else:
                    ratio = NINE_MINS/time
                new_r = old_r * ratio
                # print(f"moving {ratio}, new_r = {new_r}")
                new_point = [i, center[0] +math.cos(deg)*new_r, center[1] + math.sin(deg)*new_r]
                circle.append(new_point)
                # i, lat, long = new_point
                # deg = i/n * 2 * math.pi
                # r = (lat - center[0])/math.cos(deg)
                # print(f"new deg = {deg}, r = {r}, time {time}")
                unchanged = False
            else:
                # print("done")
                circle.append([i, lat,long])
    # print(f"iterations taken = {num_iter}")
    # for point in circle:
    #     print("(test) circle distance in km", geodesic(point, center))
    return [(lat,long) for (i, lat, long) in circle]

def has_address_in(point, city=None):
    print("DISABLED")
    return 
    res = client.reverse_geocode(point, result_type=["street_address"])
    if res == []:
        return False
    if city == None:
        return "formatted_address" in res[0]
    address = res[0]["formatted_address"]
    return city in address

def is_on_land(point):
    return globe.is_land(point[0], point[1])


# use the set covering problem matrix to just plug directly into linprog

def linprog_solver(points, matrix):
    c = np.ones(len(points))
    b = -np.ones(len(points))
    A = -matrix
    bounds = [(0,1) for _ in points]
    solution = linprog(c, A_ub=A, b_ub=b, bounds=bounds, integrality=np.ones(len(points)))
    linprog_solution = []
    for i,chosen in enumerate(solution.x):
        if round(chosen) == 1:
            linprog_solution.append(i)
    return linprog_solution

# greedy solver -> alternative to linprog to see if it makes a difference

def greedy_solve(points, matrix):
    taken = []
    internal = np.copy(matrix)
    print("Starting greedy solver")
    while internal.shape[0] != 0:
        # print(internal)
        values = np.sum(internal, axis=0)
        choice = np.argmax(values)
        taken.append(choice)
        # remove the rows that are now covered
        covered = internal[:,choice]
        to_delete = []
        for i,is_covered in enumerate(covered):
            if is_covered == 1:
                to_delete.append(i)
        internal = np.delete(internal, to_delete, axis=0)
        # remove the column for this choice
        # internal = np.delete(internal, (choice), axis=1)
    return sorted(taken)

def display_solution(center, problem, chosen_indicies, show_coverage=False):
    chosen_points = []
    for i in chosen_indicies:
        chosen_points.append(problem[i])
    
    fills = []
    if show_coverage:
        for lat,long in chosen_points:
            fills.append(distance_coverable((lat,long)))
            # break
    
    return get_map(center, markers=[("red", chosen_points), ("blue", problem)], fills=fills, zoom=11)

def equidistant_points(center, top_left, bottom_right, height=5, width=5, unit=0.015):    
    lat_max = top_left[0]
    long_min = top_left[1]
    lat_min = bottom_right[0]
    long_max = bottom_right[1]
    points = []
    for x in range(width//2):
        x_dist = distance(unit * x)
        for y in range(height//2):
            y_dist = distance(unit * y)
            for x_dir in [0, 180]:
                if x == 0 and x_dir == 180:
                    continue
                for y_dir in [90, 270]:
                    if y == 0 and y_dir == 270:
                        continue
                    p = y_dist.destination(x_dist.destination(center, bearing=x_dir), bearing=y_dir)
                    points.append((p.latitude, p.longitude))
            # points.append((center[0] - y*unit, center[1] - x*unit))
            # points.append((center[0] - y*unit, center[1] + x*unit))
            # points.append((center[0] + y*unit, center[1] - x*unit))
            # points.append((center[0] + y*unit, center[1] + x*unit))
    # print(points[:10])
    points = list(filter(lambda p: lat_min <= p[0] <= lat_max and long_min <= p[1] <= long_max, points))
    return points

def straight_line_distance_matrix(sources, destinations):
    n = len(sources)
    m = len(destinations)
    D = np.zeros((n,m))
    for i in range(n):
        start = sources[i]
        row = np.array([geodesic(start, other).kilometers for other in destinations])
        D[i:] = row
    return D

def uniform_random_evaluate_solution(hospitals, top_left = (49.295863, -123.270310), bottom_right = (49.196127, -123.021401), n=1000):
    lat_max = top_left[0]
    lat_min = bottom_right[0]
    long_max = bottom_right[1]
    long_min = top_left[1]

    # random uniform sample
    total_dist = 0
    samples = []
    for _ in range(n):
        invalid = True
        while invalid:
            lat = random.uniform(lat_min, lat_max)
            long = random.uniform(long_min, long_max)
            p = (lat, long)
            invalid = not is_on_land(p)
        distances = straight_line_distance_matrix([p], hospitals)
        best = min(distances[0])
        total_dist += best
        samples.append(p)
    return (total_dist / n, samples)