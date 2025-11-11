# A* Pathfinding Visualization

This project provides an **interactive visualization** of the **A\*** (A-Star) pathfinding algorithm using **Python** and the **Pygame** library.  
It allows users to create custom mazes by drawing barriers, setting a start and end point, and then watching the algorithm find the shortest path in real time.  
The visualization also includes a **GIF generation feature**, allowing you to save the algorithm's solution process.

---

## Overview

This project is built using Python and several key libraries:

- **Pygame** — for interactive visualization and window management.  
- **heapq** — for efficiently managing the priority queue (the "open set") in the A* algorithm.  
- **imageio** — for the GIF generation feature.

---

## Key Features

- **Interactive Grid**  
  Easily draw barriers, place the start node (blue), and set the end node (purple) with mouse clicks.

- **Real-Time Visualization**  
  Watch the algorithm as it explores the grid, visualizing the "open set" (yellow) and "closed set" (grey) in real time.

- **Weighted A\***  
  Uses a weighted heuristic to find a path faster, balancing speed with optimality.

- **GIF Export**  
  Save a GIF of the entire pathfinding process, from the search to the final path reconstruction.

---

## How to Use

- **Left Mouse Click**
  - **First Click:** Sets the **Start Node**.  
  - **Second Click:** Sets the **End Node**.  
  - **Subsequent Clicks:** Draw **Barriers** (obstacles).

- **Right Mouse Click** — Erases any node (resets it to white).  
- **Press `SPACE`** — Starts the A* algorithm.  
- **Press `R`** — Resets the entire grid.

---

## Algorithm Implementation

### The A* Algorithm

The project implements the A* algorithm, which efficiently finds the shortest path by prioritizing nodes based on the function:

$$
f(n) = g(n) + h(n)
$$

Where:

- `g(n)` — actual cost (number of steps) from the start node to the current node `n`.  
- `h(n)` — estimated cost (heuristic) from the current node `n` to the end node.

---

### Heuristic: Manhattan Distance

This implementation uses the **Manhattan Distance** for its heuristic, `h(n)`:

$$
h(n) = |x_1 - x_2| + |y_1 - y_2|
$$

It is the standard and most effective heuristic for **grid-based environments** where movement is restricted to four directions (up, down, left, right).

---

## Speeding Up the Search: Weighted A\*

This project uses a variation called **Weighted A\*** to achieve a faster solution.  
The standard formula is modified to give more weight to the heuristic:

$$
f(n) = g(n) + \mu \cdot h(n)
$$

Where `μ > 1`.  
In this code, the weight `μ` is set to **1.5**.

### Advantages

- **Benefit:** The algorithm explores significantly fewer nodes and finds a path much faster than standard A\*.  
- **Trade-off:** The resulting path is *near-optimal* — not guaranteed to be the **absolute** shortest path.

This approach is ideal for applications (like games) where finding a “good enough” path quickly is more important than finding the perfect path.

---
## Results

In this section, there are some examples of the algorithm running.

<p align="center">
  <img src="Results\A_star_solution.gif" alt="A* Search" width="45%" style="margin-right: 50px;">
  <img src="Results\A_star_solvesMaze.gif" alt="A* Solving 'Maze'" width="45%">
</p>
---

## Future Enhancements

The choice of the Manhattan Distance heuristic and the grid structure aligns perfectly with a future goal:  
**integrating a maze generation algorithm.**

Planned additions include:

- Implementation of **Kruskal’s** or **Prim’s** algorithm to dynamically generate complex mazes.  
- Allowing the A* algorithm to be tested on these newly created structures, providing a more robust test environment.

---
