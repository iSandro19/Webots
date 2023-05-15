###################################################################################################
# Robotica Q8 2022-2023
# Title: Practica 3 - Q-Learning
# Authors:
#   - Óscar Alejandro Manteiga Seoane
#   - Antonio Vila Leis
###################################################################################################
# IMPORTS 
###################################################################################################
from controller import Robot # Módulo de Webots para el control del robot

import random
from enum import Enum
import numpy as np

###################################################################################################
# CONSTANTES 
###################################################################################################
# Máxima velocidad de las ruedas soportada por el robot (khepera4).
MAX_SPEED = 47.6

# Velocidad por defecto para este comportamiento.
CRUISE_SPEED = 8

# Distancia de seguiridad a las paredes.
IR_THRESHOLD = 400

# Número de iteraciones hasta completar el aprendizaje.
ITERATIONS = 500

# Detección de la línea negra.
BLACK_THRESHOLD = 500

# Detección del espacio blanco.
WHITE_THRESHOLD = 750

# Time step por defecto para el controlador.
TIME_STEP = 16

# Tamaño del movimiento hacia adelante
SQUARE_SIZE = 25

# Radio de la rueda del robot.
WHEEL_RADIUS = 21

# Radio entre las ruedas del robot.
BETWEEN_WHEELS_RADIUS = 108.29 / 2

# Delta del incremento de la posición angular para un movimiento recto.
FORWARD_DELTA = SQUARE_SIZE / WHEEL_RADIUS

# Delta del incremento de la posición angular para un giro de (pi/4 rad).
TURN_DELTA = 0.25*np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS

# Nombres de los sensores ultrasónicos del robot.
ULTRASONIC_SENSORS = [
    "left ultrasonic sensor",
    "front left ultrasonic sensor",
    "front ultrasonic sensor",
    "front right ultrasonic sensor",
    "right ultrasonic sensor"
]

# Nombres de los sensores infrarrojos del robot.
INFRARED_SENSORS = [
    "rear left infrared sensor",
    "left infrared sensor",
    "front left infrared sensor",
    "front infrared sensor",
    "front right infrared sensor",
    "right infrared sensor",
    "rear right infrared sensor",
    "rear infrared sensor",

    "ground left infrared sensor",
    "ground front left infrared sensor",
    "ground front right infrared sensor",
    "ground right infrared sensor"
]

# Nombres de los motores
MOTOR_NAMES = [
    "left wheel motor",
    "right wheel motor",
]

# Nombres de los sensores del robot.
ENCODER_NAMES = [
    "left wheel sensor",
    "right wheel sensor",
]

###################################################################################################
# FUNCIONES AUXILIARES
###################################################################################################

# Funciones de comprobación de los sensores, estados y refuerzo. ##################################

def check_sensors(ir_sensors):
    """
    Comprueba los sensores infrarrojos del robot y devuelve una lista con los valores de los sensores.

    Args:
        ir_sensors (list): Lista con los sensores infrarrojos del robot.
    
    Returns:
        list: Lista con los valores de los sensores infrarrojos del robot.
    """

    return [ir_sensors[8].getValue(), ir_sensors[9].getValue(),
            ir_sensors[10].getValue(), ir_sensors[11].getValue()]
    
def check_estado(sensor_values, Estado):
    """
    Comprueba el estado del robot en función de los valores de los sensores infrarrojos.

    Args:
        sensor_values (list): Lista con los valores de los sensores infrarrojos del robot.
        Estado (Enum): Enumeración con los posibles estados del robot.
    
    Returns:
        Enum: Estado del robot.
    """

    if sensor_values[1] > 750 and sensor_values[3] < 500:
        return Estado.S1
    elif sensor_values[2] > 750 and sensor_values[0] < 500:
        return Estado.S2
    return Estado.S3
    
def check_refuerzo(new_sensor_values, prev_sensor_values):
    """
    Comprueba el refuerzo en función de los valores de los sensores infrarrojos.

    Args:
        new_sensor_values (list): Lista con los nuevos valores de los sensores infrarrojos del robot.
        prev_sensor_values (list): Lista con los valores anteriores de los sensores infrarrojos del robot.

    Returns:
        int: Refuerzo.
    """

    """
    # Sigue en lo negro.
    if all(value < BLACK_THRESHOLD for value in prev_sensor_values) and \
       all(value < BLACK_THRESHOLD for value in new_sensor_values):
        return 1
    
    # Sale del negro al blanco hacia la derecha.
    elif all(value < BLACK_THRESHOLD for value in prev_sensor_values) and \
         new_sensor_values[0] < BLACK_THRESHOLD and new_sensor_values[1] < BLACK_THRESHOLD and \
         new_sensor_values[2] > WHITE_THRESHOLD and new_sensor_values[3] > WHITE_THRESHOLD:
        return -0.5
    
    # Sale del negro al blanco hacia la izquierda.
    elif all(value < BLACK_THRESHOLD for value in prev_sensor_values) and \
         new_sensor_values[0] > WHITE_THRESHOLD and new_sensor_values[1] > WHITE_THRESHOLD and \
         new_sensor_values[2] < BLACK_THRESHOLD and new_sensor_values[3] < BLACK_THRESHOLD:
        return -0.5
    
    # Sale del blanco al negro hacia la derecha.
    elif prev_sensor_values[0] > WHITE_THRESHOLD and prev_sensor_values[1] > WHITE_THRESHOLD and \
         prev_sensor_values[2] < BLACK_THRESHOLD and prev_sensor_values[3] < BLACK_THRESHOLD and \
         all(value < BLACK_THRESHOLD for value in new_sensor_values):
        return 0.5
    
    # Sale del blanco al negro hacia la izquierda.
    elif prev_sensor_values[0] < BLACK_THRESHOLD and prev_sensor_values[1] < BLACK_THRESHOLD and \
         prev_sensor_values[2] > WHITE_THRESHOLD and prev_sensor_values[3] > WHITE_THRESHOLD and \
         all(value < BLACK_THRESHOLD for value in new_sensor_values):
        return 0.5
    
    # Pasa del blanco al negro completamente.
    elif all(value > WHITE_THRESHOLD for value in prev_sensor_values) and \
         all(value < BLACK_THRESHOLD for value in new_sensor_values):
        return 1
    
    # Sigue en lo blanco.
    elif all(value > WHITE_THRESHOLD for value in prev_sensor_values) and \
         all(value > WHITE_THRESHOLD for value in new_sensor_values):
        return 0
    
    # Pasa del negro al blanco completamente.
    elif all(value < BLACK_THRESHOLD for value in prev_sensor_values) and \
         all(value > WHITE_THRESHOLD for value in new_sensor_values):
        return -1
    
    # No cambiamos el refuerzo si no se cumple ningún caso.
    else:
        return 0
    """

    """
    # Todos los sensores están en negro
    if all(value < BLACK_THRESHOLD for value in new_sensor_values):
        return 1
    # Todos los sensores están en blanco
    elif all(value > WHITE_THRESHOLD for value in new_sensor_values):
        return -1
    # Mitad de los sensores están en negro
    else:
        black_sensors = sum(1 for value in new_sensor_values if value < BLACK_THRESHOLD)
        white_sensors = sum(1 for value in new_sensor_values if value > WHITE_THRESHOLD)
        if black_sensors == 1 and white_sensors == 3:
            return -1
        elif black_sensors == 2 and white_sensors == 2:
            return 0
        elif black_sensors == 3 and white_sensors == 1:
            return 1
        # No se cumple ningún caso
        return 0
    """
    if all(value < 500 for value in prev_sensor_values) and all(value < 500 for value in new_sensor_values):
        return 1
    elif all(value < 500 for value in prev_sensor_values) and not all(value < 500 for value in new_sensor_values):
        return -1
    elif all(value > 750 for value in prev_sensor_values) and all(value > 750 for value in new_sensor_values):
        return -1
    elif all(value > 750 for value in prev_sensor_values) and not all(value > 750 for value in new_sensor_values):
        return 1
    elif sum(i > 750 for i in prev_sensor_values) < sum(i > 750 for i in new_sensor_values):
        return -1
    else:
        return 1
    
# Funciones de actualización. #####################################################################

def actualizar_matriz_q(refuerzo, action, prev_estado, nuevo_estado, learning_rate, gamma_value, visitas, mat_q):
    """
    Actualiza la matriz Q conforme a la fórmula dada y al valor del refuerzo.

    Args:
        refuerzo: 
    """

    learning_rate = 1 / (1 + visitas[prev_estado.value][action])

    mat_q[prev_estado.value][action] = (1-learning_rate) * mat_q[prev_estado.value][action] + learning_rate * (refuerzo + gamma_value * np.argmax(mat_q[nuevo_estado.value]))

    visitas[prev_estado.value][action] += 1
    
    return learning_rate

# Funciones de ejecución de acciones. #############################################################
    
def pick_action(estado_actual, mat_q, cnt):
    p = cnt / ITERATIONS  # Calcula p como una función lineal de cnt
    if random.random() < p:
        print("Pensada")
        return np.argmax(mat_q[estado_actual.value])
    else:
        print("Aleatoria")
        return random.randint(0, 2)

def go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + FORWARD_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED)
        rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = rightEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and rightEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED/2)
        rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED)
        rightWheel.setVelocity(CRUISE_SPEED/2)

def perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    print(action)
    # Si la acción es 0, girar a la derecha.
    if action == 0:
        turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 1, girar a la izquierda.
    elif action == 1:
        turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 2, ir recto.
    elif action == 2:
        go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)

# Funciones de inicialización. ####################################################################

def init():
    robot = Robot()

    f_camera = robot.getDevice("camera")
    f_camera.enable(TIME_STEP)

    u_sensors = []
    for sensor in ULTRASONIC_SENSORS:
        u_sens = robot.getDevice(sensor)
        u_sens.enable(TIME_STEP)
        u_sensors.append(u_sens)

    ir_sensors = []
    for sensor in INFRARED_SENSORS:
        ir_sens = robot.getDevice(sensor)
        ir_sens.enable(TIME_STEP)
        ir_sensors.append(ir_sens)

    leftWheel = robot.getDevice(MOTOR_NAMES[0])
    rightWheel = robot.getDevice(MOTOR_NAMES[1])

    leftWheel.getPositionSensor().enable(TIME_STEP)
    rightWheel.getPositionSensor().enable(TIME_STEP)

    leftWheel.setPosition(float('inf'))
    rightWheel.setPosition(float('inf'))

    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

    leftEncoder = robot.getDevice(ENCODER_NAMES[0])
    rightEncoder = robot.getDevice(ENCODER_NAMES[1])
    
    learning_rate = 0.5
    gamma_value = 0.5

    mat_q = np.zeros((3,3))
    visitas = np.zeros((3,3))

    sensors_hist = []

    class Estado(Enum):
        S1 = 0
        S2 = 1
        S3 = 2
        
    class Accion(Enum):
        A1 = 0
        A2 = 1
        A3 = 2
        
    estado_actual = Estado.S3

    return robot, ir_sensors, leftWheel, \
           rightWheel, leftEncoder, rightEncoder, learning_rate, \
           gamma_value, mat_q, visitas, sensors_hist, \
           estado_actual, Estado, Accion

# Funciones de debug y visualización. #############################################################

def print_q_matrix(mat_q):
    # Encabezado de la matriz Q
    print("Matriz Q:")
    print("    A1   A2   A3")
    print("-----------------")

    # Mostrar la matriz Q con redondeo a 2 decimales
    for i in range(mat_q.shape[0]):
        row = f"S{i+1}: "
        for j in range(mat_q.shape[1]):
            row += f"{np.round(mat_q[i][j], 2):>5} "
        print(row)
    print("-----------------\n\n")

###################################################################################################
# MAIN
###################################################################################################
def main():
    """
    Función principal de la práctica.
    """

    robot, ir_sensors, leftWheel, \
    rightWheel, leftEncoder, rightEncoder, learning_rate, \
    gamma_value, mat_q, visitas, sensors_hist, \
    estado_actual, Estado, Accion = init()

    cnt = 0

    # Bucle principal de la simulación.
    while robot.step(TIME_STEP) != -1:
        # Lectura de sensores
        sensor_values = check_sensors(ir_sensors)
        sensors_hist.append(sensor_values)
        estado_actual = check_estado(sensor_values, Estado)

        # Realizar acción
        action = pick_action(estado_actual, mat_q, cnt)
        print(cnt)
        print_q_matrix(mat_q)
        perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    
        # Actualizar matriz Q
        sensors_hist.append(check_sensors(ir_sensors))
        new_sensor_values = sensors_hist[len(sensors_hist)-3]
        nuevo_estado = check_estado(new_sensor_values, Estado)
        refuerzo = check_refuerzo(sensor_values, new_sensor_values)
        learning_rate = actualizar_matriz_q(refuerzo, action, estado_actual, nuevo_estado, learning_rate, gamma_value, visitas, mat_q)
        
        cnt += 1

        # Comprobación de seguridad para las paredes
        while robot.step(TIME_STEP) != -1 and (ir_sensors[2].getValue() > IR_THRESHOLD or ir_sensors[3].getValue() > IR_THRESHOLD or ir_sensors[4].getValue() > IR_THRESHOLD):
            print("####! Pared detectada, ejecutando acción de seguridad !####")
            speed_offset = 0.2 * (CRUISE_SPEED - 0.03 * ir_sensors[3].getValue());
            speed_delta = 0.03 * ir_sensors[2].getValue() - 0.03 * ir_sensors[4].getValue()
            leftWheel.setVelocity(speed_offset + speed_delta)
            rightWheel.setVelocity(speed_offset - speed_delta)

if __name__ == "__main__":
    main()
