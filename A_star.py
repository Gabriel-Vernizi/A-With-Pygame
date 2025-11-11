import pygame
import math
import heapq as hq
from Spot import *

# For gif generation
import numpy as np
import imageio


WIDTH = 1200
HEIGHT = 720
WINDOW = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("A* Path Finding")


ROWS = 30
COLS = math.floor(5/3 * ROWS)

MU = 1.5

def heuristc_manhattan(p1:Spot,p2:Spot) -> (int):
    x1,y1 = p1.get_pos()
    x2,y2 = p2.get_pos()
    return abs(x1-x2) + abs(y1-y2)

def algorithm(draw,grid,start,end,mu=1,frames_list=None):
    inserted = 0
    open_set = []

    # Insert into queue values (f_score_spot,time_was_inserted,actual_spot)
    hq.heappush(open_set,(0,inserted,start))

    came_from = {}

    g_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float('inf') for row in grid for spot in row}
    f_score[start] = heuristc_manhattan(start,end)

    open_set_hash = {start}

    while len(open_set) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
        current = hq.heappop(open_set)[2] # Only the spot
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, start, draw, frames_list=frames_list)
            end.make_end()
            start.make_start()

            if frames_list is not None:
                draw(frames_list=frames_list)
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1 # Cost 1 to move to neighbor

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                # Weighted
                f_score[neighbor] = temp_g_score +  mu * heuristc_manhattan(neighbor,end)

                if neighbor not in open_set_hash:
                    inserted += 1

                    hq.heappush(open_set,(f_score[neighbor],inserted,neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw(frames_list=frames_list)
        
        if current != start:
            current.make_closed()

    return False


def reconstruct_path(came_from, end_spot, start_spot, draw, frames_list=None):
    current = came_from[end_spot]
    while current != start_spot:
        current.make_path()
        draw(frames_list=frames_list)
        current = came_from[current]


def make_grid(rows,cols,width=WIDTH,height=HEIGHT):
    grid = []
    gap_col = width//cols
    gap_rows = height//rows

    for i in range(rows):
        grid.append([])
        for j in range(cols):
            grid[i].append(Spot(i,j,gap_rows,gap_col,rows,cols))
            # print(f"spot[{i}][{j}] -> x: {i*gap_rows}, y: {j*gap_col}") # Debugging

    return grid

def draw_grid(window=WINDOW,rows=ROWS,cols=COLS,width=WIDTH,height=HEIGHT):
    gap_col = width//cols
    gap_rows = height//rows

    for i in range(rows):
        pygame.draw.line(window,GREY, start_pos=(0,i*gap_rows),end_pos=(width,i*gap_rows))

    for j in range(cols):
        pygame.draw.line(window,GREY, start_pos=(j*gap_col,0),end_pos=(j*gap_col,height))

def barrier_limit(grid,rows,cols,width=WIDTH,height=HEIGHT):
    for i in range(cols):
        grid[0][i].make_barrier()
        grid[rows-1][i].make_barrier()

    for j in range(rows):
        grid[j][0].make_barrier()
        grid[j][cols-1].make_barrier()


def draw(win,grid,rows,cols,width=WIDTH,height=HEIGHT,frames_list=None):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win,rows=rows,cols=cols,width=width,height=height)
    
    if frames_list is not None:
    # (width, height, 3) -> (height, width, 3)
        frame = np.transpose(pygame.surfarray.array3d(win), (1, 0, 2))
        frames_list.append(frame)

    pygame.display.update()

def get_clicked_pos(pos,rows,cols):


    gap_row = HEIGHT//rows
    gap_col = WIDTH//cols

    y,x = pos
    row,col = x//gap_col, y//gap_row
 
    return row,col

def main(save_gif=False):

    start = None
    end = None

    started = False

    grid = make_grid(ROWS,COLS)  
    
    pygame.init()
    clock = pygame.time.Clock()
    running = True
    
    barrier_limit(grid=grid,rows=ROWS,cols=COLS,width=WIDTH,height=HEIGHT)

    restart = 0
    while running:
        draw(WINDOW,grid,rows=ROWS,cols=COLS,width=WIDTH,height=HEIGHT)
        
        if restart:
            barrier_limit(grid=grid,rows=ROWS,cols=COLS,width=WIDTH,height=HEIGHT)
            restart = 0

        for event in pygame.event.get():
            

            if event.type == pygame.QUIT:
                print("fechou")
                running = False
            
            # Left mouse click
            if pygame.mouse.get_pressed()[0]:
                row,col = get_clicked_pos(pygame.mouse.get_pos(),ROWS,COLS)
                try:
                    spot = grid[row][col]
                except IndexError as e:
                    print(f"row: {row}, col: {col}")
                    running=False
                    raise(e)
                
                if not start:
                    start = spot
                    start.make_start()
                
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                else:
                    if spot != start and spot != end:
                        spot.make_barrier()
               
            # Right mouse click
            if pygame.mouse.get_pressed()[2]:
                row,col = get_clicked_pos(pygame.mouse.get_pos(),ROWS,COLS)
                try:
                    spot = grid[row][col]
                except IndexError as e:
                    print(f"row: {row}, col: {col}")
                    running=False
                    raise(e)

                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbor(grid)

                    if save_gif:
                        frames_para_gif = []
                    else:
                        frames_para_gif = None
                        
                    draw_gif = lambda frames_list=None: draw(
                        WINDOW, grid, ROWS, COLS, width=WIDTH, height=HEIGHT, 
                        frames_list=frames_list
                    )

                    path_found = algorithm(
                        draw_gif,
                        grid,
                        start,
                        end,
                        mu=MU,
                        frames_list=frames_para_gif
                    )

                    if save_gif and path_found:
                        print("Salvando GIF...")

                        imageio.mimsave('A_star_solution.gif', frames_para_gif, fps=30)
                        print("Sucesso! 'A_star_solution.gif' salvo na pasta.")
                
                if event.key == pygame.K_r:
                    restart = 1
                    for row in grid:
                        for spot in row:
                            spot.reset()

                    start = None
                    end = None

        clock.tick(60) 

    pygame.quit()

if __name__ == '__main__':
    main(save_gif=False)