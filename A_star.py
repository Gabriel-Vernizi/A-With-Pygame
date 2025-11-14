import pygame
import math
import heapq as hq
from Spot import Spot
import json

# For gif generation
import numpy as np
import imageio

# ------------- #

from mazes import generate_maze_prim

# ------------- #

# Colors
RED = (255, 0, 0) # Not Path
GREEN = (0, 255, 0) # Final Path
BLUE = (0, 0, 255) # Start
PURPLE = (128, 0, 128) # End
YELLOW = (255, 255, 0) # Open
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128) 
TURQUOISE = (64, 224, 208)
WHITE = (255, 255, 255) # Default
BLACK = (0, 0, 0) # Barrier

# ------------- #

WIDTH = 1200
HEIGHT = 720
WINDOW = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("A* Path Finding")


ROWS = 60
COLS = math.floor(5/3 * ROWS)

MU = 3

save_frame = 0
n_frame = 5

# ------------- #

def heuristc_manhattan(p1:Spot,p2:Spot) -> (int):
    x1,y1 = p1.get_pos()
    x2,y2 = p2.get_pos()
    return abs(x1-x2) + abs(y1-y2)

def algorithm(draw,grid,start,end,mu=1,writer=None):
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
            reconstruct_path(came_from, end, start, draw, writer=writer)
            end.make_end()
            start.make_start()

            draw(writer=writer)

            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + heuristc_manhattan(current,neighbor) # Cost 1 to move to neighbor

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

        draw(writer=writer)
        
        if current != start:
            current.make_closed()

    return False

def reconstruct_path(came_from, end_spot, start_spot, draw, writer=None):
    current = came_from[end_spot]
    while current != start_spot:
        current.make_path()
        draw(writer=writer)
        current = came_from[current]

# ------------- #

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

def draw(win,grid,rows,cols,width=WIDTH,height=HEIGHT,writer=None):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win,rows=rows,cols=cols,width=width,height=height)
    global save_frame

    if writer is not None and save_frame % n_frame == 0:

        scaled_win = pygame.transform.scale(win,(WIDTH//2,HEIGHT//2))
        frame = np.transpose(pygame.surfarray.array3d(scaled_win), (1, 0, 2))
        writer.append_data(frame)
    
    save_frame += 1

    pygame.display.update()

# ------------- #

def get_clicked_pos(pos,rows,cols):


    gap_row = HEIGHT//rows
    gap_col = WIDTH//cols

    pos_x,pos_y = pos
    
    row = pos_y//gap_row
    col = pos_x//gap_col
    return row,col
 
    return row,col

def save_last_config(grid,start,end,file='last_config.json'):
    config_data = {
            'start': start.get_pos(),
            'end': end.get_pos(),
            'barriers': []
        }

    for row in grid:
        for spot in row:
            if spot.is_barrier():
                config_data['barriers'].append(spot.get_pos())

    with open(file, 'w') as f:
        json.dump(config_data, f)
            
    print(f"Configuração salva em '{file}'")

def load_config(grid,config_file):
    with open(config_file,'r') as f:
        config_data = json.load(f)

    for i in range(1,len(grid)):
        for j in range(1,len(grid[i])):
            grid[i][j].reset()
    
    barrier_limit(grid,ROWS,COLS,WIDTH,HEIGHT)

    start_pos_x,start_pos_y = config_data['start']
    start = grid[start_pos_x][start_pos_y]
    start.make_start()

    end_pos_x,end_pos_y = config_data['end']
    end = grid[end_pos_x][end_pos_y]
    end.make_end()

    for pos in config_data['barriers']:
        grid[pos[0]][pos[1]].make_barrier()
    
    return start,end

# ------------- #

def main(save_gif=False,path_gif=r"Results/",title="A_star_solution",save_config=False):

    start = None
    end = None

    started = False

    frames_para_gif = []

    grid = make_grid(ROWS,COLS)  
    
    alternate_mu = False
    temp_mu = 1

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
                if event.key == pygame.K_SPACE:
                    
                    if not started:
                        started = 1
                    else:
                        for row in grid:
                            for spot in row:
                                if spot.is_barrier() or spot.is_start() or spot.is_end():
                                    continue
                                else:
                                    spot.reset()

                    for row in grid:
                        for spot in row:
                            spot.update_neighbor(grid)

                    if save_gif:
                        gif_path=f'{path_gif}{title}.gif'
                        writer = imageio.get_writer(gif_path,fps=60)
                    else:
                        writer = None
                        
                    draw_gif = lambda writer=None: draw(
                        WINDOW, grid, ROWS, COLS, width=WIDTH, height=HEIGHT, 
                        writer=writer
                    )

                    path_found = algorithm(
                        draw_gif,
                        grid,
                        start,
                        end,
                        mu=temp_mu,
                        writer=writer
                    )

                    if writer is not None:
                        writer.close()
                        print(f'{title}.gif salvo em {path_gif}')

                elif event.key == pygame.K_r:
                    restart = 1
                    for row in grid:
                        for spot in row:
                            spot.reset()

                    start = None
                    end = None

                elif event.key == pygame.K_g:
                    save_gif = not save_gif
                    if save_gif:
                        print('Salvamento de GIF ativado')
                    else:
                        print('Salvamento de GIF desativado')
                
                elif event.key == pygame.K_s:
                    save_config = not save_config
                    save_last_config(grid,start,end,file='last_config.json')
                    if save_config:
                        print('Salvamento de simulação feito. Salvo em ./last_config.json')
                        
                elif event.key == pygame.K_l:
                    start,end = load_config(grid,r'./last_config.json')
                
                elif event.key == pygame.K_m:
                    if save_gif:
                        gif_path=f'{path_gif}{title}.gif'
                        writer = imageio.get_writer(gif_path,fps=60)
                    else:
                        writer = None

                    draw_maze = lambda writer=None: draw(
                        WINDOW, grid, ROWS, COLS, width=WIDTH, height=HEIGHT, 
                        writer=writer
                    )

                    generate_maze_prim(draw_maze,grid,ROWS,COLS,start_pos=(1,1),writer=writer)
                    start = None
                    end = None
                
                elif event.key == pygame.K_c:
                    alternate_mu = not alternate_mu

                    if alternate_mu:
                        temp_mu = MU
                        print(f"$mu$ alterado para {temp_mu}")
                    else:
                        temp_mu = 1
                        print(f'$mu$ alterado para {temp_mu}')

                elif event.key == pygame.K_p:
                    if save_gif:
                        gif_path=f'{path_gif}{title}.gif'
                        writer = imageio.get_writer(gif_path,fps=60)
                    else:
                        writer = None

                    draw_maze = lambda writer=None: draw(
                        WINDOW, grid, ROWS, COLS, width=WIDTH, height=HEIGHT, 
                        writer=writer
                    )

                    generate_maze_prim(draw_maze,grid,ROWS,COLS,start_pos=(1,1),writer=writer)
                    start = None
                    end = None
                    
                    if writer is not None:
                        writer.close()
                        print(f'{title}.gif salvo em {path_gif}')
                    
                    break
        clock.tick(60) 

    pygame.quit()


if __name__ == '__main__':
    #main(title='A_Star_Example')
    #main(title='Weighted_A_Star_Example')

    #main(title='A_Star_Solves_Maze')
    main(title='Weighted_A_Star_Solves_Maze')
    
    #main(title="Maze_Generation")
    #main(title="Maze_Generation2")