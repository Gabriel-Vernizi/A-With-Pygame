# A* Pathfinding & Maze Generation Visualization

This project provides an **interactive visualization** of the **A\*** (A-Star) pathfinding algorithm using **Python** and the **Pygame** library.  
It allows users to create custom mazes manually or **automatically generate them** using the Randomized Prim's Algorithm. You can watch the algorithm find the shortest path in real time and save the results.

The visualization also includes a **GIF generation feature** and a system to **save/load map configurations**.

---

## Overview

This project is built using Python and several key libraries:

- **Pygame** — for interactive visualization and window management.  
- **heapq** — for efficiently managing the priority queue (the "open set") in the A* algorithm.  
- **imageio** — for the GIF generation feature.
- **NumPy** — for array manipulation during frame capture.

---

## Key Features

- **Interactive Grid** Easily draw barriers, place the start node (blue), and set the end node (purple) with mouse clicks.

- **Maze Generation (Prim's Algorithm)** Instantly generate a complex, perfect maze (no loops) using the Randomized Prim's Algorithm.

- **Real-Time Visualization** Watch the algorithm as it explores the grid, visualizing the "open set" (yellow) and "closed set" (grey) in real time.

- **Weighted A\*** Uses a weighted heuristic to find a path faster, balancing speed with optimality.

- **Save/Load System** Save your current grid layout (start, end, and barriers) to a JSON file and reload it later.

- **GIF Export** Toggle GIF recording to save the entire pathfinding process, from the search to the final path reconstruction.

---

## How to Use

### Mouse Controls
- **Left Mouse Click**
  - **First Click:** Sets the **Start Node**.  
  - **Second Click:** Sets the **End Node**.  
  - **Subsequent Clicks:** Draw **Barriers** (obstacles).
- **Right Mouse Click** — Erases any node (resets it to white).

### Keyboard Controls
- **`SPACE`** — Starts the A* algorithm.  
- **`R`** — Resets the entire grid (clears everything).  
- **`M`** — Generates a random maze using **Prim's Algorithm**.
- **`G`** — Toggles **GIF Saving** (Console will confirm "On/Off").  
- **`S`** — **Saves** the current map configuration (barriers, start, end) to `last_config.json`.
- **`L`** — **Loads** the configuration from `last_config.json`.

---

## Algorithm Implementation

### 1. The A* Algorithm

The project implements the A* algorithm, which efficiently finds the shortest path by prioritizing nodes based on the function:

$$
f(n) = g(n) + h(n)
$$

Where:
- `g(n)` — actual cost (number of steps) from the start node to the current node `n`.  
- `h(n)` — estimated cost (heuristic) from the current node `n` to the end node.

#### Heuristic: Manhattan Distance
This implementation uses the **Manhattan Distance** for its heuristic, `h(n)`:

$$
h(n) = |x_1 - x_2| + |y_1 - y_2|
$$

It is the standard and most effective heuristic for **grid-based environments** where movement is restricted to four directions.

---

### 2. Speeding Up: Weighted A\*

This project uses a variation called **Weighted A\*** to achieve a faster solution.  
The standard formula is modified to give more weight to the heuristic:

$$
f(n) = g(n) + \mu \cdot h(n)
$$

Where `μ > 1`.  
In this code, the weight `μ` is set to **2**.

**Trade-off:** The algorithm explores significantly fewer nodes and finds a path much faster, but the resulting path is *near-optimal* rather than guaranteed to be the absolute shortest.

---

### 3. Maze Generation: Randomized Prim's Algorithm

To generate mazes, the project uses the **Randomized Prim's Algorithm**. 
It behaves like a "growth" algorithm:
1. It starts with a grid full of walls.
2. It picks a random cell to be part of the maze.
3. It considers the neighbors of that cell (frontier).
4. It randomly selects a neighbor, connects it to the maze by removing the wall between them, and adds the new neighbor's neighbors to the frontier.

This results in a **Perfect Maze** (a Minimum Spanning Tree), meaning there is exactly one path between any two points and no circular loops.

---

## Results

In this section, there are some examples of the algorithm running.

<p align="center">
  <img src="Results\A_Star_Example.gif" alt="A* Search" width="45%" style="margin-right: 20px;">
  <img src="Results\Weighted_A_Star_Example.gif" alt="A* Solving 'Maze'" width="45%" style="">
</p>

<p align="center">
  <img src="Results\A_Star_Solves_Maze.gif" alt="A* Search" width="45%" style="margin-right: 20px;">
  <img src="Results\Weighted_A_Star_Solves_Maze.gif" alt="A* Solving 'Maze'" width="45%" style="">
</p>

<p align="center">
  <img src="Results\Maze_Generation.gif" alt="A* Search" width="45%" style="margin-right: 20px;">
  <img src="Results\Maze_Generation2.gif" alt="A* Solving 'Maze'" width="45%" style="">
</p>

---

## Future Enhancements

Potential future updates include:

- **UI Buttons:** Adding a graphical user interface (GUI) for buttons instead of relying solely on keyboard shortcuts.
- **Dijkstra's Algorithm:** Adding an option to run unweighted Dijkstra for comparison.

---
