import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * This class is named FinalRunner in Java.
 */
public class FinalRunner {
    private static int[][] times; // Stores the times between buildings
    private static int minTime = Integer.MAX_VALUE; // Initialize to a large value
    private static List<String> buildingNames = new ArrayList<>(); // Stores the names of the buildings
    private static List<Integer> minRoute; // Stores the minimum route
    private static ExecutorService executor = Executors.newFixedThreadPool(10); // Adjust the number of threads as needed

/**
 * The main function reads times from a file, generates routes and runs them, shuts down an executor,
 * waits for termination, and writes output to a file.
 */
    public static void main(String[] args) throws Exception {
        readTimesFromFile("input2.txt");
        for (int i = 0; i < times.length; i++) {
            generateRoutesAndRun(i, new ArrayList<>(Arrays.asList(i)), new boolean[times.length]); 
            // Generate routes
        }
        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        writeOutputToFile("output2.txt");
    }
    
/**
 * The function writes the minimum route and corresponding time to a file in Java.
 * 
 * @param filename The `filename` parameter in the `writeOutputToFile` method is a string that
 * represents the name of the file to which the output will be written. This method opens a file with
 * the specified filename and writes the output data to it.
 */
    private static void writeOutputToFile(String filename) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < minRoute.size(); i++) {
            writer.write(buildingNames.get(minRoute.get(i)) + " ");
        }
        // Add the first building at the end
        writer.write(buildingNames.get(minRoute.get(0)) + " ");
        writer.write(minTime + "\n");
        writer.close();
    }

/**
 * The function reads times from a file, parses the data, and stores it in a list of arrays.
 * 
 * @param filename The `filename` parameter in the `readTimesFromFile` method is a string that
 * represents the name of the file from which you want to read times. This method reads the times from
 * the specified file and processes them to create an array of time values.
 */
    private static void readTimesFromFile(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        List<int[]> timesList = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(":"); // Split the line by colon
            buildingNames.add(parts[0].trim()); // Add the building name
            String[] timeStrings = parts[1].trim().split(" "); // Split the time values
            int[] timeInts = new int[timeStrings.length]; // Initialize an array to store the time values
            for (int i = 0; i < timeStrings.length; i++) {
                timeInts[i] = Integer.parseInt(timeStrings[i]); // Parse the time values
            }
            timesList.add(timeInts);
        }
        reader.close();
        times = timesList.toArray(new int[0][]);
    }
    
/**
 * The function generates all possible routes and executes them using a RouteRunner.
 * 
 * @param current The `current` parameter in the `generateRoutesAndRun` method represents the current
 * node or location in the graph that is being visited during the route generation process. It is used
 * to keep track of the current position in the graph traversal algorithm.
 * @param route The `route` parameter is a list of integers representing the current route being
 * generated. It keeps track of the nodes visited in the graph so far.
 * @param visited The `visited` array is used to keep track of which nodes have been visited during the
 * route generation process. It is a boolean array where each index corresponds to a node, and `true`
 * indicates that the node has been visited, while `false` indicates that it has not been visited yet.
 */
private static void generateRoutesAndRun(int current, List<Integer> route, boolean[] visited) {
    visited[current] = true; // Mark the current building as visited
    if (route.size() == times.length) { // If all buildings have been visited
        executor.execute(new RouteRunner(new ArrayList<>(route))); // Run the route
    } else {
        for (int i = 0; i < times.length; i++) { // For each building
            if (!visited[i]) { // If the building has not been visited
                route.add(i); // Add the next building to the route
                // Calculate the time for the current route
                int time = 0;
                for (int j = 0; j < route.size() - 1; j++) {
                    time += times[route.get(j)][route.get(j + 1)];
                }
                time += times[route.get(route.size() - 1)][route.get(0)];
                if (time < minTime) { // If the time for the current route is less than the minimum time
                    generateRoutesAndRun(i, route, visited); // Recursively generate routes
                }
                route.remove(route.size() - 1); // Remove the last building from the route
            }
        }
    }
    visited[current] = false; // Mark the current building as unvisited
}

/**
 * The RouteRunner class implements the Runnable interface to calculate the time taken for a given
 * route and update the minimum time and route if necessary.
 */
    private static class RouteRunner implements Runnable {
        private List<Integer> route;

// The `RouteRunner` class is implementing the `Runnable` interface in Java. The `RouteRunner` class
// has a constructor that takes a list of integers representing a route as a parameter. When an
// instance of `RouteRunner` is created, it initializes its `route` field with the provided route.
        public RouteRunner(List<Integer> route) {
            this.route = route;
        }

        @Override
// This block of code defines the `run()` method within the `RouteRunner` class, which implements the
// `Runnable` interface. Here's what it does:
        public void run() {
            int time = 0;
            for (int i = 0; i < route.size() - 1; i++) {
                time += times[route.get(i)][route.get(i + 1)]; // Calculate the time for the route
            }
            time += times[route.get(route.size() - 1)][route.get(0)]; // Add the time to return to the starting point
            synchronized (FinalRunner.class) { // Synchronize access to shared variables
                if (time < minTime) { // Update the minimum time and route if necessary
                    minTime = time;
                    minRoute = new ArrayList<>(route);
                }
            }
        }
    } 
}