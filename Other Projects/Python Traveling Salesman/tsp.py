import numpy as np
from threading import Thread
from queue import Queue

# Define the cities and their coordinates
cities = [
    [0, 10, 1, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    [10, 0, 5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    [15, 5, 0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    [20, 15, 10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    [25, 20, 15, 5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    [30, 25, 20, 10, 5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    [35, 30, 25, 15, 10, 5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
    [40, 35, 30, 20, 15, 10, 5, 0, 5, 10, 15, 20, 25, 30, 35],
    [45, 40, 35, 25, 20, 15, 10, 5, 0, 5, 10, 15, 20, 25, 30],
    [50, 45, 40, 30, 25, 20, 15, 10, 5, 0, 5, 10, 15, 20, 25],
    [55, 50, 45, 35, 30, 25, 20, 15, 10, 5, 0, 5, 10, 15, 20],
    [60, 55, 50, 40, 35, 30, 25, 20, 15, 10, 5, 0, 5, 10, 15],
    [65, 60, 55, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, 5, 10],
    [70, 65, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, 5],
    [75, 70, 65, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
]


""" import random

cities = [[random.randint(0, 99) for _ in range(50)] for _ in range(50)]

# Ensure distance from a city to itself is 0
for i in range(50):
    cities[i][i] = 0 """

# Define a function to calculate distances between cities
def distance(city1, city2):
    return cities[city1][city2]

# Define the TSP objective function (total distance traveled)
def tsp_objective(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(route[i], route[i+1])
    return total_distance

# Define a thread to solve the TSP using nearest neighbor algorithm
class NearestNeighborThread( Thread ):
    def  __init__(self, route ): 
        Thread.__init__( self )
        self.route  = list(route)  

    def run(self):   
        new_route = [self.route[0]]   
        for _ in range(len(self.route) - 1):   
            current_city  = new_route[-1]
            remaining_cities = [city for i, city in enumerate(self.route) if i not in new_route]
            next_city_index = min(remaining_cities, key= lambda x: distance(current_city, x))
            new_route.append(next_city_index)
        self.route = new_route

    def get_route(self ):  
        return self. route

# Define a main function to solve the TSP
def solve_tsp():
    # Create threads to solve the TSP using nearest neighbor algorithm
    threads  = []
    for _ in range(10000):  # Run the nearest neighbor algorithm 10k times
        thread  = NearestNeighborThread(range(len(cities)))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Get the best route from the threads
    best_route  = min((thread.get_route() for thread in threads), key=tsp_objective)

    print("The best route is:")
    for i, city in enumerate(best_route):
        print(f"City {city}")
    print("Total distance:", tsp_objective(best_route))

# Run the main function
if __name__ == "__main__":
    solve_tsp()