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

import random # Módulo para la generación de números aleatorios
from enum import Enum # Módulo para la creación de enumeraciones
import numpy as np # Módulo de cálculo numérico

###################################################################################################
# CONSTANTES 
###################################################################################################
# Velocidad por defecto para este comportamiento.
CRUISE_SPEED = 8

# Distancia de seguiridad a las paredes.
IR_THRESHOLD = 400

# Número de iteraciones hasta completar el aprendizaje.
ITERATIONS = 1000

# Detección de la línea negra.
BLACK_THRESHOLD = 500

# Detección del espacio blanco.
WHITE_THRESHOLD = 750

# Time step por defecto para el controlador.
TIME_STEP = 16

# Tamaño del movimiento hacia adelante
SQUARE_SIZE = 10

# Radio de la rueda del robot.
WHEEL_RADIUS = 21

# Radio entre las ruedas del robot.
BETWEEN_WHEELS_RADIUS = 108.29 / 2

# Delta del incremento de la posición angular para un movimiento recto.
FORWARD_DELTA = SQUARE_SIZE / WHEEL_RADIUS

# Delta del incremento de la posición angular.
TURN_DELTA = 0.05*np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS

# Delta del incremento de la posición angular (180).
TURN_180_DELTA = np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS

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

    return [ir_sensors[8].getValue(),
            ir_sensors[9].getValue(),
            ir_sensors[10].getValue(),
            ir_sensors[11].getValue()]
    
def check_estado(sensor_values, Estado):
    """
    Comprueba el estado del robot en función de los valores de los sensores infrarrojos.

    Args:
        sensor_values (list): Lista con los valores de los sensores infrarrojos del robot.
        Estado (Enum): Enumeración con los posibles estados del robot.
    
    Returns:
        Enum: Estado del robot.
    """

    # Abandona la línea negra por la izquierda.
    if (sensor_values[0] > 750 and sensor_values[1] > 750 and sensor_values[2] > 750 and sensor_values[3] < 500) or \
       (sensor_values[0] < 500 and sensor_values[1] > 750 and sensor_values[2] < 500 and sensor_values[3] < 500) or \
       (sensor_values[0] > 750 and sensor_values[1] > 750 and sensor_values[2] < 500 and sensor_values[3] < 500):
        
        print("S1")
        return Estado.S1
    
    # Abandona la línea negra por la derecha.
    elif (sensor_values[0] < 500 and sensor_values[1] > 750 and sensor_values[2] > 750 and sensor_values[3] > 750) or \
         (sensor_values[0] < 500 and sensor_values[1] < 500 and sensor_values[2] > 750 and sensor_values[3] < 500) or \
         (sensor_values[0] < 500 and sensor_values[1] < 500 and sensor_values[2] > 750 and sensor_values[3] > 750):
         
        print("S2")
        return Estado.S2
    
    # Sigue la línea negra.
    else:
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

    if all(value < 500 for value in prev_sensor_values) and \
       all(value < 500 for value in new_sensor_values):
        return 1
    elif sum(values < 500 for values in prev_sensor_values) < sum(values < 500 for values in new_sensor_values):
        return 1
    else:
        return 0 
    
# Funciones de actualización. #####################################################################

def actualizar_matriz_q(refuerzo, action, prev_estado, nuevo_estado, learning_rate, gamma_value, visitas, mat_q):
    """
    Actualiza la matriz Q conforme a la fórmula dada y al valor del refuerzo.

    Args:
        refuerzo: 
    """

    visitas[prev_estado.value][action] += 1

    learning_rate = 1 / (1 + visitas[prev_estado.value][action])

    mat_q[prev_estado.value][action] = \
        (1-learning_rate) * mat_q[prev_estado.value][action] + \
        learning_rate * (refuerzo + gamma_value * np.argmax(mat_q[nuevo_estado.value]))
        
# Funciones de ejecución de acciones. #############################################################
    
def pick_action(estado_actual, mat_q, cnt):
    p = cnt / ITERATIONS  # Calcula p como una función lineal de cnt
    if random.random() < p:
        print("Acción pensada")
        return np.argmax(mat_q[estado_actual.value])
    else:
        print("Acción aleatoria")
        return random.randint(0, 2)

def go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + FORWARD_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED)
        rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = rightEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and rightEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(-2)
        rightWheel.setVelocity(CRUISE_SPEED + 2)
    
def turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED + 2)
        rightWheel.setVelocity(-2)

def perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    # Si la acción es 0, girar a la derecha.
    if action == 0:
        #print("Girando a la derecha")
        turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 1, girar a la izquierda.
    elif action == 1:
        #print("Girando a la izquierda")
        turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 2, ir recto.
    elif action == 2:
        #print("Yendo recto")
        go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)

# Funciones de inicialización. ####################################################################

def init():
    robot = Robot() # Crear la instancia del robot

    # Obtener los sensores de ultrasonidos
    u_sensors = []
    for sensor in ULTRASONIC_SENSORS:
        u_sens = robot.getDevice(sensor)
        u_sens.enable(TIME_STEP)
        u_sensors.append(u_sens)

    # Obtener los sensores infrarrojos
    ir_sensors = []
    for sensor in INFRARED_SENSORS:
        ir_sens = robot.getDevice(sensor)
        ir_sens.enable(TIME_STEP)
        ir_sensors.append(ir_sens)

    # Obtener los motores e inicializarlos
    leftWheel = robot.getDevice(MOTOR_NAMES[0])
    rightWheel = robot.getDevice(MOTOR_NAMES[1])
    leftWheel.getPositionSensor().enable(TIME_STEP)
    rightWheel.getPositionSensor().enable(TIME_STEP)
    leftWheel.setPosition(float('inf'))
    rightWheel.setPosition(float('inf'))
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

    # Obtener los encoders
    leftEncoder = robot.getDevice(ENCODER_NAMES[0])
    rightEncoder = robot.getDevice(ENCODER_NAMES[1])
    
    # Inicializar variables
    learning_rate = 0
    gamma_value = 0.5
    
    #mat_q = np.zeros((3,3))
    mat_q = np.identity(3)

    visitas = np.zeros((3,3))
    sensors_hist = []

    # Enumeración de estados.
    class Estado(Enum):
        S1 = 0 # Se abandona la línea por la izquierda.
        S2 = 1 # Se abandona la línea por la derecha.
        S3 = 2 # Resto de casos.
        
    # Enumeración de acciones.
    class Accion(Enum):
        A1 = 0 # Girar a la derecha.
        A2 = 1 # Girar a la izquierda.
        A3 = 2 # Ir recto.
        
    # Estado inicial.
    estado_actual = Estado.S3

    return robot, ir_sensors, leftWheel, \
           rightWheel, leftEncoder, rightEncoder, learning_rate, \
           gamma_value, mat_q, visitas, sensors_hist, \
           estado_actual, Estado, Accion

# Funciones para evitar obstáculos. ##############################################################

def avoid_obstacles(robot, ir_sensors, leftWheel, rightWheel, leftEncoder, rightEncoder):
    """
    Función para evitar obstáculos.
    """

    # Si el obstáculo está a menos de 10 cm, girar a la izquierda.
    if ir_sensors[1].getValue() > IR_THRESHOLD or ir_sensors[2].getValue() > IR_THRESHOLD or ir_sensors[3].getValue() > IR_THRESHOLD:
        initial_position = rightEncoder.getValue()
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        while robot.step(TIME_STEP) != -1 and rightEncoder.getValue() < initial_position + TURN_180_DELTA:
            leftWheel.setVelocity(-2)
            rightWheel.setVelocity(CRUISE_SPEED + 2)

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

    # Inicialización de variables obtenidas de la inicialización.
    robot, ir_sensors, leftWheel, \
    rightWheel, leftEncoder, rightEncoder, learning_rate, \
    gamma_value, mat_q, visitas, sensors_hist, \
    estado_actual, Estado, Accion = init()

    cnt = 0 # Contador de iteraciones

    # Bucle principal de la simulación.
    while robot.step(TIME_STEP) != -1:
        # Comprobación de seguridad
        print("Iteración número: ", cnt)
        # Lectura de sensores
        sensor_values = check_sensors(ir_sensors)
        sensors_hist.append(sensor_values)

        # Realizar acción
        action = pick_action(estado_actual, mat_q, cnt)
        print_q_matrix(mat_q)
        perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    
        # Actualizar matriz Q
        sensors_hist.append(check_sensors(ir_sensors))
        new_sensor_values = sensors_hist[len(sensors_hist)-3]
        nuevo_estado = check_estado(new_sensor_values, Estado)
        #refuerzo = check_refuerzo(sensor_values, new_sensor_values)
        #actualizar_matriz_q(refuerzo, action, estado_actual, nuevo_estado, learning_rate, gamma_value, visitas, mat_q)

        # Esquivar obstáculos
        #avoid_obstacles(robot, ir_sensors, leftWheel, rightWheel, leftEncoder, rightEncoder)

        estado_actual = nuevo_estado
        sensor_values
        cnt += 1 # Incrementar contador de iteraciones

if __name__ == "__main__":
    main()
