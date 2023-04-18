###################################################################################################
# Robotica Q8 2022-2023
# Title: Practica 2
# Authors:
#   - Óscar Alejandro Manteiga Seoane
#   - Antonio Vila Leis
###################################################################################################
# IMPORTS 
###################################################################################################
from controller import Robot  # Módulo de Webots para el control el robot.
from controller import Camera  # Módulo de Webots para el control de la cámara.

import time  # Si queremos utilizar time.sleep().
import numpy as np  # Si queremos utilizar numpy para procesar la imagen.
import cv2  # Si queremos utilizar OpenCV para procesar la imagen.

###################################################################################################
# CONSTANTES 
###################################################################################################
# Máxima velocidad de las ruedas soportada por el robot (khepera4).
MAX_SPEED = 47.6
# Velocidad por defecto para este comportamiento.
CRUISE_SPEED = 8
# Time step por defecto para el controlador.
TIME_STEP = 16
# Tamaño de las baldosas del suelo.
SQUARE_SIZE = 250
# Radio de la rueda del robot.
WHEEL_RADIUS = 21
# Radio entre las ruedas del robot.
BETWEEN_WHEELS_RADIUS = 108.29 / 2
# Delta del incremento de la posición angular para un movimiento recto.
FORWARD_DELTA = SQUARE_SIZE / WHEEL_RADIUS
# Delta del incremento de la posición angular para un giro de 90º (pi/2 rad).
TURN_DELTA = 0.5*np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS
# Distancia recorrida por el robot.
DISTANCE_TRAVELED = 0
# Distancia de corte a las paredes.
IR_THRESHOLD = 170
# Tamaño de las columnas de la cámara.
CAMERA_ROW_SIZE = 752
CAMERA_ROW_SIZE_2 = CAMERA_ROW_SIZE // 2
CAMERA_ROW_SIZE_4 = CAMERA_ROW_SIZE // 4
# Tamaño de las filas de la cámara.
CAMERA_COL_SIZE = 480
CAMERA_COL_SIZE_2 = CAMERA_COL_SIZE // 2
CAMERA_COL_SIZE_4 = CAMERA_COL_SIZE // 4

# Número de canales de la cámara.
CAMERA_COLOR_CHANNELS = {
    "red": 0,
    "green": 1,
    "blue": 2,
}

# Nombres de los sensores de distancia basados en infrarrojo.
INFRARED_SENSORS_NAMES = [
    "rear left infrared sensor",
    "left infrared sensor",
    "front left infrared sensor",
    "front infrared sensor",
    "front right infrared sensor",
    "right infrared sensor",
    "rear right infrared sensor",
    "rear infrared sensor",
]

# Nombres de los motores
MOTOR_NAMES = [
    "left wheel motor",
    "right wheel motor",
]

# Nombres de los sensores de posición de las ruedas.
ENCODER_NAMES = [
    "left wheel sensor",
    "right wheel sensor",
]

# Nombres de los posibles estados del robot.
STATES = {
    "INIT": 0,
    "MAP_MAKING": 1,
    "PATROL": 2,
    "COME_BACK": 3,
    "STOP": 4,
}

# Tipos de movimientos.
MOVEMENTS = {
    "FORWARD": 0,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 2,
}

###################################################################################################
# FUNCIONES AUXILIARES
###################################################################################################

# Funciones para el control de sensores/actuadores. ###############################################

def enable_distance_sensors(robot, timeStep, sensorNames):
    """
    Obtener y activar los sensores de distancia.

    robot: objeto robot de Webots.
    timeStep: tiempo (en milisegundos) de actualización por defecto para los sensores/actuadores
        (cada dispositivo puede tener un valor diferente).
    sensorNames: lista con los nombres de los sensores de distancia a activar.

    Return: lista con los sensores de distancia activados, en el mismo orden
    establecido en la lista de  nombres (sensorNames).
    """

    sensorList = []

    for name in sensorNames:
        sensorList.append(robot.getDevice(name))

    for sensor in sensorList:
        sensor.enable(timeStep)

    return sensorList

def init_devices(timeStep):
    """
    Obtener y configurar los dispositivos necesarios.

    timeStep: tiempo (en milisegundos) de actualización por defecto para los sensores/actuadores
      (cada dispositivo puede tener un valor diferente).
    """

    # Get pointer to the robot.
    robot = Robot()

    # Si queremos obtener el timestep de la simulación.
    # simTimeStep = int(robot.getBasicTimeStep())

    # Obtener dispositivos correspondientes a los motores de las ruedas.
    leftWheel = robot.getDevice(MOTOR_NAMES[0])
    rightWheel = robot.getDevice(MOTOR_NAMES[1])

    # Configuración inicial para utilizar movimiento por posición (necesario para odometría).
    # En movimiento por velocidad, establecer posición a infinito (wheel.setPosition(float('inf'))).
    leftWheel.setPosition(0)
    rightWheel.setPosition(0)
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

    # Obtener una lista con los sensores infrarrojos ya activados
    irSensorList = enable_distance_sensors(robot, timeStep, INFRARED_SENSORS_NAMES)

    # Obtener el dispositivo de la cámara
    camera = robot.getDevice("camera")
    # Activar el dispositivo de la cámara (el tiempo de actualización de los frames
    # de la cámara no debería ser muy alto debido al alto volumen de datos que implica).
    camera.enable(timeStep * 10)

    # Obtener y activar los sensores de posición de las ruedas (encoders).
    posL = robot.getDevice(ENCODER_NAMES[0])
    posR = robot.getDevice(ENCODER_NAMES[1])
    posL.enable(timeStep)
    posR.enable(timeStep)

    return robot, leftWheel, rightWheel, irSensorList, posL, posR, camera

def process_image_hsv(camera):
    """
    Procesamiento del último frame capturado por el dispositivo de la cámara
    utilizando el espacio de color RGB para una detección de color.

    camera: dispositivo de la cámara.
    """

    # Obtener la imagen capturada por la cámara.
    image = camera.getImage()
    # Convertir la imagen a un array de numpy.
    image = np.array(camera.getImageArray())
    
    # Hacer la media de los valores de los píxeles de la imagen en el rango de
    # la zona de interés (ROI) para cada canal de color.
    red = int(image[
        CAMERA_ROW_SIZE_4:(CAMERA_ROW_SIZE_4 + CAMERA_ROW_SIZE_2),
        CAMERA_COL_SIZE_4:(CAMERA_COL_SIZE_4 + CAMERA_COL_SIZE_2),
        CAMERA_COLOR_CHANNELS.get("red")
        ].mean())
    green = int(image[
        CAMERA_ROW_SIZE_4:(CAMERA_ROW_SIZE_4 + CAMERA_ROW_SIZE_2),
        CAMERA_COL_SIZE_4:(CAMERA_COL_SIZE_4 + CAMERA_COL_SIZE_2),
        CAMERA_COLOR_CHANNELS.get("green")
        ].mean())
    blue = int(image[
        CAMERA_ROW_SIZE_4:(CAMERA_ROW_SIZE_4 + CAMERA_ROW_SIZE_2),
        CAMERA_COL_SIZE_4:(CAMERA_COL_SIZE_4 + CAMERA_COL_SIZE_2),
        CAMERA_COLOR_CHANNELS.get("blue")
        ].mean())

    if (red >= 200 and green >= 200 and blue <= 100):
        return True
    else:
        return False

# Funciones para la gestión del mapa y las rutas del robot. ######################################

def init_map():
    """
    Inicializa una matriz 27x27 celdas llena de ceros.
    """
    return [[0] * 27 for _ in range(27)]

def update_map(current_pos_map, map):
    """
    Actualiza la matriz del mapa del entorno.

    current_pos_map: tupla que indica la posición actual del robot en la matriz.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    irSensorList: lista con los sensores de distancia infrarrojos activados.
    map: matriz que representa el mapa del entorno.
    """
    # Actualizar la celda actual como visitada.
    map[current_pos_map[0]][current_pos_map[1]] = 1

    print("- \n\n")

    for row in map:
        print(" ".join("." if cell == 0 else "#" for cell in row))

def check_goal(current_pos_map, direction, initial_direction):
    """
    Comprueba si el robot ha llegado al objetivo.

    current_pos_map: tupla que indica la posición actual del robot en la matriz.
    direction: entero que indica la dirección actual del robot.
    initial_direction: entero que indica la dirección inicial del robot.
    """

    if ((initial_direction - direction) % 4) == 0 and current_pos_map == (13, 13):
        return True
    
def heuristic(a, b):
    """
    Función heurística para el algoritmo A*. Distancia de Manhattan.

    a: nodo inicial.
    b: nodo final.
    """
    
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def create_graph(map):
    """
    Crea un grafo no dirigido a partir de un mapa.

    map: matriz que representa el mapa.

    Devuelve un diccionario que representa el grafo.
    """
    
    graph = {}

    for row in range(len(map)):
        for col in range(len(map[0])):
            if not map[row][col]:
                continue

            node = (row, col)
            neighbors = []
            if row > 0 and map[row - 1][col]:
                neighbors.append((row - 1, col))
            if row < len(map) - 1 and map[row + 1][col]:
                neighbors.append((row + 1, col))
            if col > 0 and map[row][col - 1]:
                neighbors.append((row, col - 1))
            if col < len(map[0]) - 1 and map[row][col + 1]:
                neighbors.append((row, col + 1))

            graph[node] = neighbors

    return graph

def a_star(map, start, goal):
    """
    Algoritmo A* para encontrar el camino más corto entre dos celdas de un mapa.

    map: matriz que representa el mapa.
    start: celda inicial.
    goal: celda final.
    """

    graph = create_graph(map)

    open_list = [start]
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        current = min(open_list, key=lambda x: f_score[x])
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        open_list.remove(current)
        closed_set.add(current)

        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in open_list or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_list:
                    open_list.append(neighbor)

    return open_list

def move_to_cell(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map, initial_pos, robot, path):
    """
    Mueve el robot a lo largo del camino especificado por la lista de celdas 'path'.

    leftWheel: dispositivo del motor de la rueda izquierda.
    rightWheel: dispositivo del motor de la rueda derecha.
    encoderL: dispositivo del sensor de posición de la rueda izquierda.
    encoderR: dispositivo del sensor de posición de la rueda derecha.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    current_pos_map: tupla que indica la posición actual del robot en la matriz.
    path: lista de tuplas que representan las celdas del camino a seguir.
    """
    
    for cell in path:
        stopped = True
        exit_cell = False
        check = True
        direction_factor = 0

        # Moverse a la celda de abajo.
        if cell[0] > current_pos_map[0]:
            direction_factor = 0

        # Moverse a la celda de arriba.
        elif cell[0] < current_pos_map[0]:
            direction_factor = 2

        # Moverse a la celda de la derecha.
        elif cell[1] > current_pos_map[1]:
            direction_factor = 1

        # Moverse a la celda de la izquierda.
        elif cell[1] < current_pos_map[1]:
            direction_factor = 3
        else:
            check = False
        
        if check:
            # Ejecución del movimiento.
            if direction % 4 == direction_factor:
                    exit_cell = True

            while robot.step(TIME_STEP) != -1 and not exit_cell:
                if stopped:
                    direction = turn_left(leftWheel, rightWheel, encoderL, encoderR, direction)
                    initial_pos = encoderL.getValue()
                    stopped = False

                if encoderL.getValue() <= initial_pos - TURN_DELTA + 0.005:
                    stop(leftWheel, rightWheel)
                    stopped = True

                    if direction % 4 == direction_factor:
                        exit_cell = True

            stopped = True
            exit_cell = False

            while robot.step(TIME_STEP) != -1 and not exit_cell:
                if stopped:
                    current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                    initial_pos = encoderL.getValue()
                    stopped = False

                if encoderL.getValue() >= initial_pos + FORWARD_DELTA:
                    stop(leftWheel, rightWheel)
                    stopped = True
                
                    if current_pos_map == cell:
                        exit_cell = True

    stop(leftWheel, rightWheel)


# Funciones de debug. #############################################################################

def printIrSensors(irSensorList):
    """
    Imprime los valores de los sensores infrarrojos.

    irSensorList: lista de dispositivos de los sensores infrarrojos.
    """

    print("Valores de los sensores infrarrojos:")
    for i, sensor in enumerate(irSensorList):
        print("Sensor {}: {}".format(i, sensor.getValue()))

# Funciones de movimiento. ########################################################################

def stop(leftWheel, rightWheel):
    """
    Para el robot.

    leftWheel: dispositivo del motor de la rueda izquierda.
    rightWheel: dispositivo del motor de la rueda derecha.
    """

    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)
     
def forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map):
    """
    Mueve el robot hacia delante.

    leftWheel: dispositivo del motor de la rueda izquierda.
    rightWheel: dispositivo del motor de la rueda derecha.
    encoderL: dispositivo del sensor de posición de la rueda izquierda.
    encoderR: dispositivo del sensor de posición de la rueda derecha.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    current_pos_map: tupla que indica la posición actual del robot en la matriz.
    """

    if direction % 4 == 0:
        current_pos_map = (current_pos_map[0] + 1, current_pos_map[1])
    elif direction % 4 == 1:
        current_pos_map = (current_pos_map[0], current_pos_map[1] + 1)
    elif direction % 4 == 2:
        current_pos_map = (current_pos_map[0] - 1, current_pos_map[1])
    elif direction % 4 == 3:
        current_pos_map = (current_pos_map[0], current_pos_map[1] - 1)

    incrementL = FORWARD_DELTA
    incrementR = FORWARD_DELTA
    leftWheel.setVelocity(CRUISE_SPEED) 
    rightWheel.setVelocity(CRUISE_SPEED)
    leftWheel.setPosition(encoderL.getValue() + incrementL)
    rightWheel.setPosition(encoderR.getValue() + incrementR)
    
    return current_pos_map
    
def turn_left(leftWheel, rightWheel, encoderL, encoderR, direction):
    """
    Gira el robot hacia la izquierda.

    leftWheel: dispositivo del motor de la rueda izquierda.
    rightWheel: dispositivo del motor de la rueda derecha.
    encoderL: dispositivo del sensor de posición de la rueda izquierda.
    encoderR: dispositivo del sensor de posición de la rueda derecha.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    """

    direction += 1
    leftWheel.setVelocity(CRUISE_SPEED) 
    rightWheel.setVelocity(CRUISE_SPEED)  
    leftWheel.setPosition(encoderL.getValue()-TURN_DELTA) 
    rightWheel.setPosition(encoderR.getValue()+TURN_DELTA)
    return direction
    
def turn_right(leftWheel, rightWheel, encoderL, encoderR, direction):
    """
    Gira el robot hacia la derecha.

    leftWheel: dispositivo del motor de la rueda izquierda.
    rightWheel: dispositivo del motor de la rueda derecha.
    encoderL: dispositivo del sensor de posición de la rueda izquierda.
    encoderR: dispositivo del sensor de posición de la rueda derecha.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    """

    direction -= 1
    leftWheel.setVelocity(CRUISE_SPEED) 
    rightWheel.setVelocity(CRUISE_SPEED)  
    leftWheel.setPosition(encoderL.getValue()+TURN_DELTA) 
    rightWheel.setPosition(encoderR.getValue()-TURN_DELTA)
    return direction

###################################################################################################
# MAIN
###################################################################################################
def main():
    """
    Función principal.
    """

    # Activamos los dispositivos necesarios y obtenemos referencias a ellos.
    robot, leftWheel, rightWheel, irSensorList, encoderL, encoderR, camera = init_devices(TIME_STEP)

    # Variables para el control del robot.
    direction = 0
    movement = -1
    states = 0
    stopped = True

    initial_direction = 0
    initial_pos = encoderL.getValue()
    current_pos_map = (13, 13)

    # Inicializar el mapa.
    map = init_map()

    # Bucle inicial.
    while robot.step(TIME_STEP) != -1 and STATES.get("INIT") == states:
        if stopped:
            if irSensorList[1].getValue() < IR_THRESHOLD:
                    direction = turn_left(leftWheel, rightWheel, encoderL, encoderR, direction)
                    
                    initial_pos = encoderL.getValue()
                    stopped = False
                    movement = MOVEMENTS.get("TURN_LEFT")
            else:
                states = STATES.get("MAP_MAKING")
                initial_direction = direction

        if encoderL.getValue() <= initial_pos - TURN_DELTA + 0.01 and movement == MOVEMENTS.get("TURN_LEFT"):
            # Parar el robot.
            stopped = True
            stop(leftWheel, rightWheel)

    # Bucle principal.
    while robot.step(TIME_STEP) != -1 and (STATES.get("MAP_MAKING") == states or STATES.get("PATROL") == states):
        if stopped:
            # Si hay pared a la izquierda, seguir de frente
            if irSensorList[1].getValue() > IR_THRESHOLD and irSensorList[3].getValue() < IR_THRESHOLD:
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")

            # Si no hay pared de frente y acabamos de girar, seguir de frente
            elif irSensorList[3].getValue() < IR_THRESHOLD and movement == MOVEMENTS.get("TURN_LEFT"):
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")
                
            # Si hay pared de frente, girar a la derecha
            elif irSensorList[3].getValue() > IR_THRESHOLD and irSensorList[1].getValue() > IR_THRESHOLD:
                direction = turn_right(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_RIGHT")
                
            # Si no hay pared a la izquierda, girar a la izquierda
            else:
                direction = turn_left(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_LEFT")
                
        # Comprobar si el robot ha llegado a la posición objetivo.
        if  encoderL.getValue() >= initial_pos + FORWARD_DELTA and movement == MOVEMENTS.get("FORWARD") or \
            encoderL.getValue() <= initial_pos - TURN_DELTA + 0.005 and movement == MOVEMENTS.get("TURN_LEFT") or \
            encoderL.getValue() >= initial_pos + TURN_DELTA - 0.005 and movement == MOVEMENTS.get("TURN_RIGHT"):
            
            # Parar el robot.
            stopped = True
            stop(leftWheel, rightWheel)

            # Procesar la imagen.
            if process_image_hsv(camera) and STATES.get("PATROL") == states:
                states = STATES.get("COME_BACK")

        # Actualizar mapa y comprobar si el robot ha llegado a la posición objetivo.
        if encoderL.getValue() >= initial_pos + FORWARD_DELTA and movement == MOVEMENTS.get("FORWARD"):
            # Comprobar si hay un obstáculo en cada dirección y actualizar el mapa.
            update_map(current_pos_map, map)
        
        # Comprobar si el robot ha llegado a la posición objetivo.
        if check_goal(current_pos_map, direction, initial_direction):
            states = STATES.get("PATROL")

    # Bucle de vuelta a casa.
    if robot.step(TIME_STEP) != -1 and STATES.get("COME_BACK") == states:
        path = a_star(map, current_pos_map, (13, 13))
        move_to_cell(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map, initial_pos, robot, path)

if __name__ == "__main__":
    main()

