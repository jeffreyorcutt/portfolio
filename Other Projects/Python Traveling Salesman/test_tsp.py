import unittest
from tsp import distance, tsp_objective, NearestNeighborThread, cities

class TestTSPMethods(unittest.TestCase):

    def test_distance(self):
        self.assertEqual(distance(0, 1), 10)
        self.assertEqual(distance(1, 2), 5)
        self.assertEqual(distance(3, 4), 5)

class TestTSPMethods(unittest.TestCase):

    def test_tsp_objective(self):
        # Test a route that goes from city 0 to 1 to 2 and back to 0
        self.assertEqual(tsp_objective([0, 1, 2, 0]), 30)

        # Test a route that goes from city 0 to 2 to 1 and back to 0
        self.assertEqual(tsp_objective([0, 2, 1, 0]), 30)

        # Test a route that goes from city 0 to 14 to 13 and back to 0
        self.assertEqual(tsp_objective([0, 14, 13, 0]), 150)


    def test_nearest_neighbor_thread(self):
        thread = NearestNeighborThread(range(len(cities)))
        thread.start()
        thread.join()
        route = thread.get_route()
        self.assertEqual(len(route), len(cities))  # The route should visit all cities
        self.assertEqual(len(set(route)), len(cities))  # The route should not visit any city twice

if __name__ == '__main__':
    unittest.main()