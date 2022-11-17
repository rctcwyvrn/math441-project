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

api_key = open("googlemaps-apikey.txt", "r").read()
client = googlemaps.Client(key=api_key)

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
        if chosen == 1:
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
        shift = False
        for y in range(height//2):
            shift = not shift
            # if shift:
            #     center = (center[0] - unit/2, center[1])
            # else:
            #     center = (center[0] + unit/2, center[1])
            points.append((center[0] - y*unit, center[1] - x*unit))
            points.append((center[0] - y*unit, center[1] + x*unit))
            points.append((center[0] + y*unit, center[1] - x*unit))
            points.append((center[0] + y*unit, center[1] + x*unit))
    # print(points[:10])
    points = list(filter(lambda p: lat_min <= p[0] <= lat_max and long_min <= p[1] <= long_max, points))
    return points