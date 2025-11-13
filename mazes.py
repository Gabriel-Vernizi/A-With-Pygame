import random


def generate_maze_prim(draw, grid, rows, cols,start_pos = (1,1)):


    for row in grid:
        for spot in row:
            spot.make_barrier()
    
    draw() # Reset

    frontier = []

    start_row, start_col = start_pos

    def add_frontier(r, c):
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 < nr < rows-1 and 0 < nc < cols-1:
                neighbor = grid[nr][nc]
                parent = grid[r][c]

                if neighbor.is_barrier():
                    frontier.append((neighbor, parent))

    add_frontier(start_row, start_col)

    while frontier:

        rand_index = random.randint(0, len(frontier) - 1)
        current_spot, parent_spot = frontier.pop(rand_index)

        if current_spot.is_barrier():
            current_spot.reset()

            # Current pos
            c_row, c_col = current_spot.get_pos()
            # Parent pos
            p_row, p_col = parent_spot.get_pos()
            
            
            wall_row = (c_row + p_row) // 2
            wall_col = (c_col + p_col) // 2
            
            grid[wall_row][wall_col].reset()

            
            add_frontier(c_row, c_col)
            
            draw()

    # Atualiza vizinhos para o A* funcionar depois
    for row in grid:
        for spot in row:
            spot.update_neighbor(grid)
