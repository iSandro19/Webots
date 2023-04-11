from controller import Robot, Motor, DistanceSensor
import numpy as np
from collections import deque

TIME_STEP = 32
MAX_SPEED = 5
SQUARE_SIZE = 250
WHEEL_RADIUS = 21
FORWARD_ONE = SQUARE_SIZE / WHEEL_RADIUS
TURN_VAL = 4.05

robot = Robot()

leftWheel = robot.getDevice("left wheel motor")
rightWheel = robot.getDevice("right wheel motor")
leftWheel.getPositionSensor().enable(TIME_STEP)
rightWheel.getPositionSensor().enable(TIME_STEP)
encoderL = robot.getDevice("left wheel sensor")
encoderR = robot.getDevice("right wheel sensor") 

posL = encoderL.getValue() 
posR = encoderR.getValue()

l_ir_sensor = robot.getDevice("left infrared sensor")
r_ir_sensor = robot.getDevice("right infrared sensor")
f_ir_sensor = robot.getDevice("front infrared sensor")
l_ir_sensor.enable(TIME_STEP)
r_ir_sensor.enable(TIME_STEP)
f_ir_sensor.enable(TIME_STEP)

f_camera = robot.getDevice("camera")
f_camera.enable(TIME_STEP)

found_goal = False

stopped = 0

map = np.ones((27,27))
map[0],map[-1],map[:,0],map[:,-1] = 3,3,3,3

INITIAL_POS = (13,13)
current_pos_map = INITIAL_POS
direction = 0


# 0-Vacio, 1-No explorado, 2-Visitado, 3-Pared
def mark_pos(pos, ir):
    if ir > 190:
        map[pos] = 3
    elif map[pos]==1:
        map[pos] = 0
    
def mark_near(pos, direction, ir):
    if direction%4 == 0:
        mark_pos((pos[0],pos[1]+1), ir[0]) # Left
        mark_pos((pos[0],pos[1]-1), ir[1]) # Right
        mark_pos((pos[0]+1,pos[1]), ir[2]) # Fwd
    elif direction%4 == 1:
        mark_pos((pos[0]-1,pos[1]), ir[0]) # Left
        mark_pos((pos[0]+1,pos[1]), ir[1]) # Right
        mark_pos((pos[0],pos[1]+1), ir[2]) # Fwd
    elif direction%4 == 2:
        mark_pos((pos[0],pos[1]-1), ir[0]) # Left
        mark_pos((pos[0],pos[1]+1), ir[1]) # Right
        mark_pos((pos[0]-1,pos[1]), ir[2]) # Fwd
    elif direction%4 == 3:
        mark_pos((pos[0]+1,pos[1]), ir[0]) # Left
        mark_pos((pos[0]-1,pos[1]), ir[1]) # Right
        mark_pos((pos[0],pos[1]-1), ir[2]) # Fwd
        
def decide_dir(pos, dir, movement, map):
    if dir%4 == 0:
        array_directions = [
            map[(pos[0],pos[1]+1)],
            map[(pos[0]+1,pos[1])],
            map[(pos[0],pos[1]-1)]
        ] # Left, fwd, right
    elif dir%4 == 1:
        array_directions = [
            map[(pos[0]-1,pos[1])],
            map[(pos[0],pos[1]+1)],
            map[(pos[0]+1,pos[1])]
        ] # Left, fwd, right
    elif dir%4 == 2:
        array_directions = [
            map[(pos[0],pos[1]-1)],
            map[(pos[0]-1,pos[1])],
            map[(pos[0],pos[1]+1)]
        ] # Left, fwd, right
    elif dir%4 == 3:
        array_directions = [
            map[(pos[0]+1,pos[1])],
            map[(pos[0],pos[1]-1)],
            map[(pos[0]-1,pos[1])]
        ] # Left, fwd, right
    #if array_directions[0]==array_directions[2] and array_directions[0]<array_directions[1]:
    #    return 2
    index = np.argmin(array_directions)
    #print(array_directions)
    if all(element == array_directions[0] for element in array_directions) and array_directions[0]==3:
        return 2
    elif movement == 1 and index==0:
        return index + 1
    return index
        
def get_nearest_empty(pos,map):
    aux=np.asarray(np.where(map==0))
    near = None
    min_dist = None
    for x in range(aux.shape[1]):
        zero_loc = (aux[:,x])
        dist = np.sqrt(pow((pos[0] - zero_loc[0]), 2) + pow(pos[1] - zero_loc[1], 2))
        if min_dist is None or dist<min_dist:
            min_dist = dist
            near = (zero_loc[0],zero_loc[1])
    return near
    
def bfs(start,goal,map):
    queue = deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (y,x) == goal:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < 27 and 0 <= y2 < 27 and map[x2][y2] != 3 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
                
def create_path(path, map):
    for cell in path:
        map[cell]=0

def stop_robot():
    leftWheel.setVelocity(0) 
    rightWheel.setVelocity(0)
    
def move_fwd(current_pos_map, direction):
    if direction%4 == 0:
        current_pos_map = (current_pos_map[0] + 1, current_pos_map[1])
    elif direction%4 == 1:
        current_pos_map = (current_pos_map[0], current_pos_map[1] + 1)
    elif direction%4 == 2:
        current_pos_map = (current_pos_map[0] - 1, current_pos_map[1])
    elif direction%4 == 3:
        current_pos_map = (current_pos_map[0], current_pos_map[1] - 1)
    incrementL = FORWARD_ONE
    incrementR = FORWARD_ONE
    leftWheel.setVelocity(MAX_SPEED) 
    rightWheel.setVelocity(MAX_SPEED)
    leftWheel.setPosition(encoderL.getValue() + incrementL)
    rightWheel.setPosition(encoderR.getValue() + incrementR)
    return current_pos_map
    
def measure_ir():
    front_add_value = 0
    if check_goal():
        front_add_value = 200
    return[l_ir_sensor.getValue(),
        r_ir_sensor.getValue(),
        f_ir_sensor.getValue() + front_add_value]
        
def turn_left(direction):
    stop_robot()
    direction += 1
    leftWheel.setVelocity(MAX_SPEED) 
    rightWheel.setVelocity(MAX_SPEED)  
    leftWheel.setPosition(encoderL.getValue()-TURN_VAL) 
    rightWheel.setPosition(encoderR.getValue()+TURN_VAL)
    return direction
    
def turn_right(direction):
    stop_robot()
    direction -= 1
    leftWheel.setVelocity(MAX_SPEED) 
    rightWheel.setVelocity(MAX_SPEED)  
    leftWheel.setPosition(encoderL.getValue()+TURN_VAL) 
    rightWheel.setPosition(encoderR.getValue()-TURN_VAL)
    return direction
    

def check_goal():
    image = np.array(f_camera.getImageArray())
    red = int(image[266:486,72:408,0].mean())
    green = int(image[266:486,72:408,1].mean())
    blue = int(image[266:486,72:408,2].mean())
    #print("R: "+str(red)+" G: "+str(green)+" B: "+str(blue))
    if (red <= 110 and green >= 120 and blue <= 80):
        return True
    else:
        return False
        
def check_completed(pos, moved):
    return pos==INITIAL_POS and moved==1

def mark_object_as_wall(pos, dir):
    if dir%4 == 0:
        map[(pos[0]+1,pos[1])]
    elif dir%4 == 1:
        map[(pos[0],pos[1]+1)]
    elif dir%4 == 2:
        map[(pos[0]-1,pos[1])]
    elif dir%4 == 3:
        map[(pos[0],pos[1]-1)]

    
stop_robot()

leftWheel.setVelocity(MAX_SPEED) 
rightWheel.setVelocity(MAX_SPEED)

initial_pos = encoderL.getValue()

movement = 0
mapping = 1
has_moved = 0
initial_direction = 0

finished_mapping = False
finished_searching = False
finished_returning = False

print("Started mapping")

# Comportamiento 1: mapeado
while robot.step(TIME_STEP) != -1 and finished_mapping == False:
    current_pos = encoderL.getValue()
    if (movement == 0 and current_pos - initial_pos >= FORWARD_ONE):
        stopped = 0
        has_moved = 1
    elif (movement == 1 and initial_pos - current_pos >= TURN_VAL -0.004):
        stopped = 0
    elif (movement == 2 and current_pos - initial_pos >= TURN_VAL -0.004):
        stopped = 0
    if stopped == 0:
        finished_mapping = check_completed(current_pos_map, has_moved)
        ir_values = measure_ir()
        mark_near(current_pos_map, direction, ir_values)
        if has_moved:
            map[current_pos_map]=2
        else:
            map[current_pos_map]=0
        if not finished_mapping:
            next_dir = decide_dir(current_pos_map, direction, movement, map) # 0-Left, 1-Fwd, 2-Right
            if(next_dir == 0 and has_moved == 1):
                movement = 1
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_left(direction)
                if not has_moved:
                    initial_direction = direction        
            elif(next_dir == 1):
                movement = 0
                stopped = 1
                initial_pos = encoderL.getValue()
                current_pos_map = move_fwd(current_pos_map, direction)
            elif(next_dir == 2 or (next_dir == 0 and has_moved == 0)):
                movement = 2
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_right(direction)
                if not has_moved:
                    initial_direction = direction
                    
print("Finished mapping")
print("Started searching")
map=np.where((map==0)|(map==2),1,map)
print(map)
# Comportamiento 2: buscar el objeto
while robot.step(TIME_STEP) != -1 and finished_searching == False:
    current_pos = encoderL.getValue()
    if (movement == 0 and current_pos - initial_pos >= FORWARD_ONE):
        stopped = 0
        has_moved = 1
    elif (movement == 1 and initial_pos - current_pos >= TURN_VAL -0.004):
        stopped = 0
    elif (movement == 2 and current_pos - initial_pos >= TURN_VAL -0.004):
        stopped = 0
    if stopped == 0:
        finished_searching = check_goal()
        if not finished_searching:
            print(current_pos_map)
            next_dir = decide_dir(current_pos_map, direction, movement, map) # 0-Left, 1-Fwd, 2-Right
            if(next_dir == 0 and has_moved == 1):
                movement = 1
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_left(direction)
                if not has_moved:
                    initial_direction = direction        
            elif(next_dir == 1):
                movement = 0
                stopped = 1
                initial_pos = encoderL.getValue()
                current_pos_map = move_fwd(current_pos_map, direction)
            elif(next_dir == 2 or (next_dir == 0 and has_moved == 0)):
                movement = 2
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_right(direction)
                if not has_moved:
                    initial_direction = direction

print("Finished searching")
print("Started returning")

#Comportamiento 3: volver

#We need to mark the object as a wall, in order to be able to find a suitable path to return.
#Otherwise, the robot may find a path through the object and get stuck as a result.
#mark_object_as_wall(current_pos_map,direction)

#Calculating the returning path
print(current_pos_map)
print(INITIAL_POS)
print(map)
path_to_return = bfs(current_pos_map,INITIAL_POS, map)
#print("PATH TO RETURN: " + path_to_return)

create_path(path_to_return, map)

print(map)
while robot.step(TIME_STEP) != -1 and finished_returning == False:
    current_pos = encoderL.getValue()
    if (movement == 0 and current_pos - initial_pos >= FORWARD_ONE):
        stopped = 0
        has_moved = 1
    elif (movement == 1 and initial_pos - current_pos >= TURN_VAL -0.004):
        stopped = 0
    elif (movement == 2 and current_pos - initial_pos >= TURN_VAL -0.004):
        stopped = 0
    if stopped == 0:
        finished_returning = current_pos_map == INITIAL_POS
        if not finished_returning:
            next_dir = decide_dir(current_pos_map, direction, movement, map) # 0-Left, 1-Fwd, 2-Right
            if next_dir == -1:
                nearest = get_nearest_empty(current_pos_map, map)
                path=bfs(current_pos_map,nearest, map)
                create_path(path, map)
            if(next_dir == 0 and has_moved == 1):
                movement = 1
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_left(direction)
                if not has_moved:
                    initial_direction = direction        
            elif(next_dir == 1):
                movement = 0
                stopped = 1
                initial_pos = encoderL.getValue()
                current_pos_map = move_fwd(current_pos_map, direction)
            elif(next_dir == 2 or (next_dir == 0 and has_moved == 0)):
                movement = 2
                stopped = 1
                initial_pos = encoderL.getValue()
                direction = turn_right(direction)
                if not has_moved:
                    initial_direction = direction