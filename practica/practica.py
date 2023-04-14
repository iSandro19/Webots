###################################################################################################
# Robotica Q8 2022-2023
# Title: Practica 2
# Authors:
#   - Óscar Alejandro Manteiga Seoane
#   - Antonio Vila Leis
###################################################################################################
# Imports 
###################################################################################################
from controller import Robot  # Módulo de Webots para el control el robot.
from controller import Camera  # Módulo de Webots para el control de la cámara.

import time  # Si queremos utilizar time.sleep().
import numpy as np  # Si queremos utilizar numpy para procesar la imagen.
import cv2  # Si queremos utilizar OpenCV para procesar la imagen.

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
IR_THRESHOLD = 185
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
    "MAP_MAKING": 0,
    "PATROL": 1,
    "COME_BACK": 2,
    "STOP": 3,
}

# Tipos de movimientos.
MOVEMENTS = {
    "FORWARD": 0,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 2,
}

###################################################################################################
# Funciones auxiliares.
###################################################################################################
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
        print("Amarillo!")

def init_map():
    """
    Inicializa una matriz 12x12 celdas llena de ceros.
    """
    return [[0] * 12 for _ in range(12)]

def update_map(current_pos_map, direction, irSensorList, map):
    """
    Actualiza la matriz del mapa del entorno.

    current_pos_map: tupla que indica la posición actual del robot en la matriz.
    direction: entero que indica la dirección actual del robot (0=norte, 1=este, 2=sur, 3=oeste).
    irSensorList: lista con los sensores de distancia infrarrojos activados.
    map: matriz que representa el mapa del entorno.
    """
    # Actualizar la celda actual como visitada.
    map[current_pos_map[0]][current_pos_map[1]] = 1

    for row in map:
        print(" ".join("." if cell == 0 else "#" for cell in row))

    """
    Versión para detectar paredes (no usada)

    # Comprobar si hay un obstáculo en cada dirección y actualizar el mapa.
    for i, sensor in enumerate(irSensorList):
        if i not in [1, 3, 5]:
            continue
        sensor_value = sensor.getValue()
        if sensor_value < IR_THRESHOLD:
            if direction == 0:
                map[current_pos_map[0] - 1][current_pos_map[1]] = 2  # obstáculo al norte
            elif direction == 1:
                map[current_pos_map[0]][current_pos_map[1] + 1] = 2  # obstáculo al este
            elif direction == 2:
                map[current_pos_map[0] + 1][current_pos_map[1]] = 2  # obstáculo al sur
            elif direction == 3:
                map[current_pos_map[0]][current_pos_map[1] - 1] = 2  # obstáculo al oeste

    for row in map:
        print(" ".join("." if cell == 0 else "#" for cell in row))
    """

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

def main():
    """
    Función principal.
    """

    # Activamos los dispositivos necesarios y obtenemos referencias a ellos.
    robot, leftWheel, rightWheel, irSensorList, encoderL, encoderR, camera = init_devices(TIME_STEP)

    # Variables para el control del robot.
    direction = 0
    movement = -1
    stopped = True
    initial_pos = encoderL.getValue()
    current_pos_map = (0, 3)

    # Inicializar el mapa.
    map = init_map()

    while robot.step(TIME_STEP) != -1:       
        if stopped:
            # Si hay pared a la izquierda, seguir de frente
            if irSensorList[1].getValue() > IR_THRESHOLD and irSensorList[3].getValue() < IR_THRESHOLD:
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")
                
                print("Movimiento hacia adelante")

            elif irSensorList[3].getValue() < IR_THRESHOLD and movement == MOVEMENTS.get("TURN_LEFT"):
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")
                
                print("Movimiento hacia adelante")

            elif irSensorList[3].getValue() > IR_THRESHOLD:
                direction = turn_right(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_RIGHT")
                
                print("Giro a la derecha")

            # Si no hay pared a la izquierda, girar a la izquierda
            else:
                direction = turn_left(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_LEFT")
                
                print("Giro a la izquierda")

        if  encoderL.getValue() >= initial_pos + FORWARD_DELTA and movement == MOVEMENTS.get("FORWARD") or \
            encoderL.getValue() <= initial_pos - TURN_DELTA + 0.01 and movement == MOVEMENTS.get("TURN_LEFT") or \
            encoderL.getValue() >= initial_pos + TURN_DELTA - 0.01 and movement == MOVEMENTS.get("TURN_RIGHT"):
            
            # Parar el robot.
            stopped = True
            stop(leftWheel, rightWheel)

            # Procesar la imagen y obtener la posición actual del robot en el mapa.
            process_image_hsv(camera)

        if encoderL.getValue() >= initial_pos + FORWARD_DELTA and movement == MOVEMENTS.get("FORWARD"):
            print("Posición actual: " + str(current_pos_map))
            # Comprobar si hay un obstáculo en cada dirección y actualizar el mapa.
            update_map(current_pos_map, direction, irSensorList, map)

if __name__ == "__main__":
    main()