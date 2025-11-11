import pygame

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

class Spot:
    def __init__(self,row,col,spot_height,spot_width,total_rows,total_cols):
        self.row = row;
        self.col = col;

        self.spot_height = spot_height;
        self.spot_width = spot_width;

        self.x = col * spot_width;
        self.y = row * spot_height;

        self.color = WHITE;  
        self.neighbors = [];
        self.total_rows = total_rows;
        self.total_cols = total_cols;
    
    def get_pos(self):
        return self.row,self.col;
    
    def get_pos_grid(self):
        return self.x,self.y;

    # Get Status
    def is_closed(self):
        return self.color == RED;

    def is_open(self):
        return self.color == WHITE;

    def is_barrier(self):
        return self.color == BLACK;

    def is_start(self):
        return self.color == BLUE;

    def is_end(self):
        return self.color == PURPLE;

    # Change Status

    def reset(self):
        self.color = WHITE;
    
    def make_open(self):
        self.color = YELLOW;
    
    def make_path(self):
        self.color = GREEN;

    def make_closed(self):
        self.color = GREY;

    def make_barrier(self):
        self.color = BLACK;

    def make_start(self):
        self.color = BLUE;

    def make_end(self):
        self.color = PURPLE;

    # Draw 
    def draw(self,window):
        pygame.draw.rect(window,self.color,(self.x,self.y,self.spot_width,self.spot_height))

    # The algorithm doesn't search in diagonal, i.e, we only need to focus on adjacent neighbors
    def update_neighbor(self,grid):
        self.neighbors = []
        # spot[x-1][y]
        if self.col > 0 and not (grid[self.row][self.col-1].is_barrier()):
            self.neighbors.append(grid[self.row][self.col-1])
        # spot[x+1][y]
        if self.col < self.total_cols - 1 and not (grid[self.row][self.col+1].is_barrier()):
            self.neighbors.append(grid[self.row][self.col+1])
        # spot[x][y-1]
        if self.row > 0 and not (grid[self.row-1][self.col].is_barrier()):
            self.neighbors.append(grid[self.row-1][self.col])
        # spot[x][y+1]
        if self.row < self.total_rows -1 and not (grid[self.row + 1][self.col].is_barrier()):
            self.neighbors.append(grid[self.row+1][self.col])

        return
    
    def color_neighbors(self,grid,COLOR):
        for spot in self.neighbors:
            spot.color = COLOR


    def __lt__(self,other):
        return False
    
    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}"
    
