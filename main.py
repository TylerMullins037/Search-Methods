import csv
import math
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import heapq
import tracemalloc

route_data_collections = []

# Function to load city data from a CSV file
def load_cities(file_path):
    cities = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name, lat, lon = row[0], float(row[1]), float(row[2])
            cities[name] = (lat, lon)
    return cities

# Function to load adjacency data from a CSV file
def load_adjacencies(file_path, cities):
    adjacencies = defaultdict(list)
    distances = {}

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            city_a, city_b = row[0].strip().split(' ')
            adjacencies[city_a].append(city_b)
            adjacencies[city_b].append(city_a)  # Ensure symmetry

            # Calculate distance and store it
            lat_a, lon_a = cities[city_a]
            lat_b, lon_b = cities[city_b]
            dist = distance(lat_a, lon_a, lat_b, lon_b)
            distances[(city_a, city_b)] = dist
            distances[(city_b, city_a)] = dist  # Symmetric distance

    return adjacencies, distances

def distance(lat1, lon1, lat2, lon2): # Function to calculate the distance between two points on Earth
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2]) # Convert to radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r    
    
def plot_cities(cities, adjacencies, distances, route=None, traversal_edges=None):
    # Get the dimensions of the Tkinter window
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    
    # Adjust the figure size
    fig, ax = plt.subplots(figsize=(window_width / 100, window_height / 100))  # Convert pixels to inches

    # Plot cities
    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, 'bo')  # Plot city point
        ax.text(lon, lat, city, fontsize=8, ha='right')  # Label the city

    # Plot connections and distances
    for city, neighbors in adjacencies.items():
        for neighbor in neighbors:
            lat1, lon1 = cities[city]
            lat2, lon2 = cities[neighbor]
            ax.plot([lon1, lon2], [lat1, lat2], 'r-', zorder=1)  # Red lines for connections

    # Highlight traversal path in yellow (only for real existing connections in BFS traversal order)
    if traversal_edges:
        for city_a, city_b in traversal_edges:
            if city_b in adjacencies[city_a]:  # Ensure this is a valid connection
                lat_a, lon_a = cities[city_a]
                lat_b, lon_b = cities[city_b]
                ax.plot([lon_a, lon_b], [lat_a, lat_b], 'yellow', linewidth=2, zorder=2)

    # Highlight final route in green (if a route is found)
    if route:
        for i in range(len(route) - 1):
            city_a, city_b = route[i], route[i + 1]
            if city_b in adjacencies[city_a]:  # Ensure this is a valid connection
                lat_a, lon_a = cities[city_a]
                lat_b, lon_b = cities[city_b]
                ax.plot([lon_a, lon_b], [lat_a, lat_b], 'green', linewidth=2, zorder=3)

    plt.title('City Connections with Traversal and Route')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()

    return fig

def reconstruct_path(predecessors, start, goal):
    # Reconstruct the path from start to goal using the predecessors
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessors.get(current)
    path.reverse()  # Reverse to get the correct order
    return path if path[0] == start else None  # Return None if start is not in path

def breadth_first_search(start, goal, adjacencies, distances):
    queue = deque([(start, [start], 0)])  # (current_city, path, cost)
    visited = set([start])  # Start by marking the start city as visited
    min_costs = {start: 0}  # Track minimum costs to reach each city
    traversal_path = []
    traversal_edges = []
    predecessors = {start: None}

    while queue:
        current_city, path, current_cost = queue.popleft()
        traversal_path.append(current_city)

        # If we reach the goal, reconstruct the path and return
        if current_city == goal:
            return traversal_path, reconstruct_path(predecessors, start, goal), traversal_edges

        # Explore neighboring cities
        for neighbor in adjacencies[current_city]:
            # Calculate the cost to the neighbor
            edge_cost = distances.get((current_city, neighbor), float('inf'))
            new_cost = current_cost + edge_cost

            # Check if we have not visited this neighbor or found a cheaper path
            if neighbor not in visited or new_cost < min_costs.get(neighbor, float('inf')):
                visited.add(neighbor)  # Mark as visited
                min_costs[neighbor] = new_cost  # Update minimum cost
                predecessors[neighbor] = current_city  # Record predecessor
                queue.append((neighbor, path + [neighbor], new_cost))  # Enqueue the neighbor
                traversal_edges.append((current_city, neighbor))  # Record the edge

        # Mark the current city as fully explored only after exploring all neighbors
        visited.add(current_city)

    return traversal_path, None, traversal_edges 
 
 # Depth-First Search algorithm for route finding
def depth_first_search(start, goal, adjacencies):
    stack = [(start, [start])]  # Use stack for DFS (current_city, path)
    visited = set()
    traversal_path = []
    traversal_edges = []

    while stack:
        current_city, path = stack.pop()

        if current_city == start:
            traversal_edges.append((start, current_city))
        else: 
            traversal_edges.append((current_city, prev_city))
        traversal_path.append(current_city)
        visited.add(current_city)
        
        if current_city == goal:
            return traversal_path, path, traversal_edges 
        
        for neighbor in adjacencies[current_city]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))  # Add to stack for further exploration
        prev_city = current_city
    return traversal_path, None, traversal_edges 

# Helper function to perform Depth-Limited Search up to a certain depth limit
def depth_limited_search(city, goal, adjacencies, depth, traversal_path, path, traversal_edges):
    traversal_path.append(city)
    if depth == 0 and city == goal:
        return True
    if depth > 0:
        for neighbor in adjacencies[city]:
            if neighbor not in traversal_path:  # Avoid cycles by skipping visited cities
                traversal_edges.append((city, neighbor))
                path.append(neighbor)
                if depth_limited_search(neighbor, goal, adjacencies, depth - 1, traversal_path, path, traversal_edges):
                    return True
                path.pop()  # Backtrack
    return False

# Iterative Deepening Depth-First Search (IDDFS) algorithm for route finding
def iddfs(start, goal, adjacencies, max_depth=50):
    for depth in range(max_depth):
        traversal_path = []
        path = [start]
        traversal_edges = []
        if depth_limited_search(start, goal, adjacencies, depth, traversal_path, path, traversal_edges):
            return traversal_path, path, traversal_edges
    return traversal_path, None, traversal_edges

def heuristic(current_city, goal_city, cities):
    """Calculate the straight-line distance between current city and goal city (heuristic function)."""
    lat1, lon1 = cities[current_city]
    lat2, lon2 = cities[goal_city]
    return distance(lat1, lon1, lat2, lon2)

def best_first_search(start, goal, adjacencies, cities):
    # Priority queue (min-heap) initialized with the start city, a heuristic value of 0, and the path containing just the start
    pq = [(0,start,[start])]
    visited = set()
    traversal_edges = []
    traversal_path = []
    came_from = {}

    while pq:
        # Pop the city with the lowest heuristic value (h_value) from the priority queue
        h_value, current_city, path = heapq.heappop(pq)
        traversal_path.append(current_city)
        visited.add(current_city)   

        if current_city != start:
            traversal_edges.append((came_from[current_city], current_city))

        # If the goal city is reached, return the exploration order, the path to the goal, and the traversal edges
        if current_city == goal:
            return traversal_path,path,traversal_edges
        
        # Explore each neighboring city of the current city
        for neighbor in adjacencies[current_city]:
            if neighbor not in visited:
                visited.add(neighbor)
                h_neighbor = heuristic(neighbor,goal,cities) # Compute the heuristic value for the neighbor
                # Push the neighbor onto the priority queue with its heuristic value, the neighbor itself, and the updated path
                heapq.heappush(pq,(h_neighbor, neighbor, path +[neighbor]))
                came_from[neighbor] = current_city
    return traversal_path, None, traversal_edges

def a_star_search(start, goal, adjacencies, cities):
    open_set = []
    heapq.heappush(open_set, (0, start, [start]))  # (f(n), node, path)
    closed_set = set()
    g = {start: 0}
    traversal_edges = []
    traversal_path = []
    came_from = {}

    while open_set:
        _, current_city, path = heapq.heappop(open_set)
        traversal_path.append(current_city)

        if current_city != start:
            traversal_edges.append((came_from[current_city], current_city))
        if current_city == goal:
            return traversal_path, path, traversal_edges
        
        closed_set.add(current_city)

        for neighbor in adjacencies[current_city]:
            if neighbor in closed_set:
                continue
            
             # Calculate the tentative g-value (cost from the start to this neighbor)
            tentative_g = g[current_city] + distances[(current_city, neighbor)]

            # If this path to the neighbor is shorter or if the neighbor hasn't been explored yet
            if neighbor not in g or tentative_g < g[neighbor]:
                g[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal, cities)  # Calculate the f-value (g + heuristic h)
                # Add the neighbor to the open set with its f-value, updated path, and position in the graph
                heapq.heappush(open_set, (f, neighbor, path + [neighbor]))
                came_from[neighbor] = current_city
        
            

    return traversal_path, None, traversal_edges

def cost_calc(route, traversal_edges, distances):
    traversal_cost = 0
    route_cost = 0
    
    # Calculate the cost for the traversal edges
    for city_a, city_b in traversal_edges:
        if (city_a, city_b) in distances:
            traversal_cost += distances[(city_a, city_b)]
    
    # Calculate the cost for the final route
    for i in range(len(route) - 1):
        city_a, city_b = route[i], route[i + 1]
        if (city_a, city_b) in distances:
            route_cost += distances[(city_a, city_b)]

    return traversal_cost, route_cost

def compare_algorithms():
    start_city = start_city_var.get()
    goal_city = goal_city_var.get()

    if start_city == "" or goal_city == "" or start_city == goal_city:
        messagebox.showerror("Input Error", "Please select unique starting and ending cities.")
        return
    
    # Define the algorithms and their corresponding functions
    algorithms = {
        "Breadth First Search": lambda: breadth_first_search(start_city, goal_city, adjacencies, distances),
        "Depth First Search": lambda: depth_first_search(start_city, goal_city, adjacencies),
        "Iterative Deepening DFS": lambda: iddfs(start_city, goal_city, adjacencies),
        "Best-First Search": lambda: best_first_search(start_city, goal_city, adjacencies, cities),
        "A* Search": lambda: a_star_search(start_city, goal_city, adjacencies, cities)
    }
    
    route_data_collections.clear()  # Clear previous comparison data

    # Loop through each algorithm, run it, and store the results
    for algo_name, algo_func in algorithms.items():
        sum_time = 0
        tracemalloc.start()
        for i in range(1000):
            start_time = time.perf_counter()
            traversal, route, traversal_edges = algo_func()
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000
            sum_time += elapsed_time

        current, peak = tracemalloc.get_traced_memory()
        memory_usage = current / 10**6
        tracemalloc.stop()
        elapsed_time_avg = sum_time / 1000

        # Calculate costs
        if route:
            traversal_cost, route_cost = cost_calc(route, traversal_edges, distances)
        else:
            traversal_cost, route_cost = float('inf'), float('inf')

        # Store results in the collection
        route_data_collections.append([algo_name, traversal_cost, route_cost, elapsed_time_avg, memory_usage])

    # Display the comparison results in a new window
    display_comparison_results()

def display_comparison_results():
    # Create a new window to display the comparison
    comparison_window = tk.Toplevel(root)
    comparison_window.title("Algorithm Comparison Results")

    # Create a treeview for displaying comparison data
    columns = ('Algorithm', 'Traversal Cost (km)', 'Route Cost (km)', 'Time Elapsed (ms)', 'Memory Usage (MB)')
    tree = ttk.Treeview(comparison_window, columns=columns, show='headings')

    # Define column headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)  # Set a fixed width for each column

    # Insert comparison data
    for data in route_data_collections:
        tree.insert('', tk.END, values=data)

    # Pack the treeview into the window
    tree.pack(expand=True, fill='both')

    # Add a button to close the window
    close_button = tk.Button(comparison_window, text="Close", command=comparison_window.destroy)
    close_button.pack(pady=10)

def find_route():
    start_city = start_city_var.get()
    goal_city = goal_city_var.get()
    selected_algorithm = algorithm_var.get()

    if start_city == "" or goal_city == "" or start_city == goal_city:
        messagebox.showerror("Input Error", "Please select unique starting and ending cities.")
        return
    if selected_algorithm == "":
        messagebox.showerror("Input Error", "Please select a valid algorithm.")
        return
    
    sum = 0
    tracemalloc.start()
    if selected_algorithm == "Compare All":
            compare_algorithms()
    for i in range(1000):
        start_time = time.perf_counter()
        # Perform selected algorithm to get the traversal and route information
        if selected_algorithm == "Breadth First Search":
            traversal, route, traversal_edges = breadth_first_search(start_city, goal_city, adjacencies,distances)
        elif selected_algorithm == "Depth First Search":
            traversal, route, traversal_edges = depth_first_search(start_city, goal_city, adjacencies)
        elif selected_algorithm == "Iterative Deepening DFS":
            traversal, route, traversal_edges = iddfs(start_city, goal_city, adjacencies)
        elif selected_algorithm == "Best-First Search":
            traversal, route, traversal_edges = best_first_search(start_city, goal_city, adjacencies, cities)
        elif selected_algorithm == "A* Search":
            traversal, route, traversal_edges = a_star_search(start_city, goal_city, adjacencies, cities)
        
            
        
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        sum += elapsed_time 

    current, peak = tracemalloc.get_traced_memory()
    memory_usage = current/(10**6)
    tracemalloc.stop()
   
    elasped_time = sum/1000

    if route:
        # Clear existing plots from the canvas
        for widget in frame_map.winfo_children():
            widget.destroy()

        # Plot traversal edges and the route
        fig = plot_cities(cities, adjacencies, distances, route=route, traversal_edges=traversal_edges)

        # Create a fixed-size canvas
        canvas = FigureCanvasTkAgg(fig, master=frame_map)
        canvas.draw()

        # Set the size of the canvas (example size, adjust as needed)
        canvas.get_tk_widget().config(width=1300, height=750)
        canvas.get_tk_widget().pack()

        # Display traversal and route in the listbox
        traversal_list.delete(0, tk.END)
        route_list.delete(0, tk.END)
        traversal_list.insert(tk.END, *traversal)
        route_list.insert(tk.END, *route)

        traversal_cost, route_cost = cost_calc(route, traversal_edges, distances)
        route_data_collections.append([start_city,goal_city,selected_algorithm, traversal_cost,route_cost,elapsed_time,memory_usage])
        # Display the costs in the labels
        traversal_cost_label.config(text=f"Traversal Cost: {traversal_cost:.2f} km")
        route_cost_label.config(text=f"Route Cost: {route_cost:.2f} km")

        elapsed_time_label.config(text=f"Time Elapsed: {elapsed_time:.4f} ms")
        memory_usage_label.config(text=f"Memory Usage: {memory_usage:.6f} MB") 
        # Hide the Find Route button after clicking
        find_route_button.pack_forget()
        root.update()
    else:
        messagebox.showinfo("No Route Found", f"No route found between {start_city} and {goal_city}.")

def reset_map():
    # Clear the canvas and listboxes
    for widget in frame_map.winfo_children():
        widget.destroy()
    traversal_list.delete(0, tk.END)
    route_list.delete(0, tk.END)

    traversal_cost_label.config(text="Traversal Cost: N/A")
    route_cost_label.config(text="Route Cost: N/A")
    elapsed_time_label.config(text="Time Elapsed: N/A")
    memory_usage_label.config(text="Memory Usage: N/A")
    # Show the Find Route button again
    find_route_button.pack()
    
# Load city and adjacency data
cities = load_cities("Coordinates.csv")
adjacencies, distances = load_adjacencies("Adjacencies.txt", cities)

# Create GUI
root = tk.Tk()
root.title("City Route Finder")

# Create frames
frame_map = tk.Frame(root, width=1300, height=750)
frame_map.grid(row=0, column=0, rowspan=6)


frame_controls = tk.Frame(root)
frame_controls.grid(row=0, column=1, padx=10, pady=10)

frame_traversal = tk.Frame(root)
frame_traversal.grid(row=2, column=1, padx=10, pady=10)

frame_route = tk.Frame(root)
frame_route.grid(row=4, column=1, padx=10, pady=10)

# Dropdowns for start and goal cities
start_city_var = tk.StringVar()
goal_city_var = tk.StringVar()
algorithm_var = tk.StringVar()  # Default selection

start_city_label = tk.Label(frame_controls, text="Start City:")
start_city_label.pack()
start_city_dropdown = ttk.Combobox(frame_controls, textvariable=start_city_var, values=list(cities.keys()))
start_city_dropdown.pack()

goal_city_label = tk.Label(frame_controls, text="Goal City:")
goal_city_label.pack()
goal_city_dropdown = ttk.Combobox(frame_controls, textvariable=goal_city_var, values=list(cities.keys()))
goal_city_dropdown.pack()

# Dropdown for selecting the algorithm
algorithm_label = tk.Label(frame_controls, text="Select Algorithm:")
algorithm_label.pack()
algorithm_dropdown = ttk.Combobox(frame_controls, textvariable=algorithm_var, values=["Breadth First Search","Depth First Search", "Iterative Deepening DFS", "Best-First Search","A* Search"])  # Add more algorithms as you implement them
algorithm_dropdown.pack()

# Button to find route
find_route_button = tk.Button(frame_controls, text="Find Route", command=find_route)
find_route_button.pack(pady=5)

# Button to reset map
reset_button = tk.Button(frame_controls, text="Reset", command=reset_map)
reset_button.pack(pady=5)

# Traversal path list
traversal_label = tk.Label(frame_traversal, text="Traversal Path:")
traversal_label.pack()
traversal_list = tk.Listbox(frame_traversal)
traversal_list.pack()

traversal_cost_label = tk.Label(frame_controls, text="Traversal Cost: N/A")
traversal_cost_label.pack(pady=5)

elapsed_time_label = tk.Label(frame_controls, text="Time Elapsed: N/A")
elapsed_time_label.pack(pady=5)

memory_usage_label = tk.Label(frame_controls, text="Memory Usage: N/A")
memory_usage_label.pack(pady=5)

route_cost_label = tk.Label(frame_controls, text="Route Cost: N/A")
route_cost_label.pack(pady=5)

compare_button = tk.Button(frame_controls, text="Compare All", command=compare_algorithms)
compare_button.pack(pady=5)

# Final route list
route_label = tk.Label(frame_route, text="Final Route:")
route_label.pack()
route_list = tk.Listbox(frame_route)
route_list.pack()

# Start the GUI loop
root.mainloop()
