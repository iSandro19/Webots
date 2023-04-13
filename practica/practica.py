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
    utilizando el espacio de color HSV para una detección de color.

    RECOMENDACIÓN: utilizar OpenCV para procesar más eficientemente la imagen.
    """

    # Obtener el último frame capturado por la cámara.
    image = camera.getImage()

    # Convertir la imagen RGB a HSV.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Definir los límites inferior y superior del rango de color que queremos detectar (en este caso, rojo).
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Crear una máscara para detectar el color rojo en la imagen HSV.
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Aplicar un filtro morfológico de apertura para eliminar pequeñas imperfecciones en la máscara.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Buscar los contornos en la máscara para identificar objetos de color rojo.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si se han encontrado contornos, dibujar un rectángulo alrededor del objeto más grande.
    if len(contours) > 0:
        # Encontrar el contorno más grande.
        largest_contour = max(contours, key=cv2.contourArea)
        # Encontrar las coordenadas del rectángulo que encierra el contorno más grande.
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Dibujar un rectángulo alrededor del objeto más grande.
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar la imagen resultante en una ventana.
    cv2.imshow("Processed Image", image)
    cv2.waitKey(1)

def stop(leftWheel, rightWheel):
    """
    Para el robot.
    """

    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)
     
def forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map):
    """
    Mueve el robot hacia delante.
    """

    if direction%4 == 0:
        current_pos_map = (current_pos_map[0] + 1, current_pos_map[1])
    elif direction%4 == 1:
        current_pos_map = (current_pos_map[0], current_pos_map[1] + 1)
    elif direction%4 == 2:
        current_pos_map = (current_pos_map[0] - 1, current_pos_map[1])
    elif direction%4 == 3:
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
    """

    direction -= 1
    leftWheel.setVelocity(CRUISE_SPEED) 
    rightWheel.setVelocity(CRUISE_SPEED)  
    leftWheel.setPosition(encoderL.getValue()+TURN_DELTA) 
    rightWheel.setPosition(encoderR.getValue()-TURN_DELTA)
    return direction

def main():
    # Activamos los dispositivos necesarios y obtenemos referencias a ellos.
    robot, leftWheel, rightWheel, irSensorList, encoderL, encoderR, camera = init_devices(TIME_STEP)

    # Variables para el control del robot.
    direction = 0
    movement = -1
    stopped = True
    initial_pos = encoderL.getValue()
    current_pos_map = (0, 0)

    while robot.step(TIME_STEP) != -1:       
        if stopped:
            # Si hay pared a la izquierda, seguir de frente
            if irSensorList[1].getValue() > 185 and irSensorList[3].getValue() < 185:
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")
                
                print("Frente")

            elif irSensorList[3].getValue() < 185 and movement == MOVEMENTS.get("TURN_LEFT"):
                current_pos_map = forward(leftWheel, rightWheel, encoderL, encoderR, direction, current_pos_map)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("FORWARD")
                
                print("Frente")

            elif irSensorList[3].getValue() > 185:
                direction = turn_right(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_RIGHT")
                
                print("Derecha")

            # Si no hay pared a la izquierda, girar a la izquierda
            else:
                direction = turn_left(leftWheel, rightWheel, encoderL, encoderR, direction)
                
                initial_pos = encoderL.getValue()
                stopped = False
                movement = MOVEMENTS.get("TURN_LEFT")
                
                print("Izquierda")

        if encoderL.getValue() >= initial_pos + FORWARD_DELTA and movement == MOVEMENTS.get("FORWARD"):
            stopped = True
            stop(leftWheel, rightWheel)
            print("Parado (después de ir hacia delante)")
        elif encoderL.getValue() <= initial_pos - TURN_DELTA + 0.01 and movement == MOVEMENTS.get("TURN_LEFT"):
            stopped = True
            stop(leftWheel, rightWheel)
            print("Parado (después de girar a la izquierda)")
        elif encoderL.getValue() >= initial_pos + TURN_DELTA - 0.01 and movement == MOVEMENTS.get("TURN_RIGHT"):
            stopped = True
            stop(leftWheel, rightWheel)
            print("Parado (después de girar a la derecha)")

if __name__ == "__main__":
    main()