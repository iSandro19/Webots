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
IR_THRESHOLD = 500

# Número de iteraciones hasta completar el aprendizaje.
ITERATIONS = 1000

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

# Delta del incremento de la posición angular.
TURN_DELTA = 0.1*np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS

# Delta del incremento de la posición angular (aprx 180).
TURN_180_DELTA = 0.5*np.pi * BETWEEN_WHEELS_RADIUS / WHEEL_RADIUS

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

    if sensor_values[1] > 750 and sensor_values[3] < 500:
        print("Estado S1 (salida por la izquierda)")
        return Estado.S1
    elif sensor_values[2] > 750 and sensor_values[0] < 500:
        print("Estado S2 (salida por la derecha)")
        return Estado.S2
    else:
        print("Estado S3 (resto de casos)")
        return Estado.S3
    
def check_refuerzo(prev_sensor_values, new_sensor_values):
    """
    Comprueba el refuerzo en función de los valores de los sensores infrarrojos.

    Args:
        prev_sensor_values (list): Lista con los valores anteriores de los sensores infrarrojos del robot.
        new_sensor_values (list): Lista con los valores nuevos de los sensores infrarrojos del robot.

    Returns:
        int: Refuerzo.
    """

    prev_black_sensors = sum(i < 500 for i in prev_sensor_values)
    new_black_sensors = sum(i < 500 for i in new_sensor_values)

    if new_black_sensors == 4:
        return 0.5
    return new_black_sensors - prev_black_sensors
    
# Funciones de actualización. #####################################################################

def actualizar_matriz_q(refuerzo, action, prev_estado, nuevo_estado, learning_rate, gamma_value, visitas, mat_q):
    """
    Actualiza la matriz Q conforme a la fórmula dada y al valor del refuerzo.

    Args:
        refuerzo: Refuerzo.
        action: Acción.
        prev_estado: Estado anterior.
        nuevo_estado: Nuevo estado.
        learning_rate: Tasa de aprendizaje.
        gamma_value: Factor de descuento.
        visitas: Matriz de visitas.
        mat_q: Matriz Q.S 
    """

    learning_rate = 1 / (1 + visitas[prev_estado.value][action])

    mat_q[prev_estado.value][action] = \
        (1-learning_rate) * mat_q[prev_estado.value][action] + \
        learning_rate * (refuerzo + gamma_value * np.argmax(mat_q[nuevo_estado.value]))
    
    visitas[prev_estado.value][action] += 1
        
# Funciones de ejecución de acciones. #############################################################
    
def pick_action(estado_actual, mat_q, cnt):
    """
    Elige la acción a realizar en función del estado actual y la matriz Q.

    Args:
        estado_actual: Estado actual.
        mat_q: Matriz Q.
        cnt: Contador de iteraciones.

    Returns:
        int: Acción a realizar.    
    """
    
    p = cnt / ITERATIONS  # Calcula p como una función lineal de cnt
    if random.random() < p:
        print("Realizando acción óptima")
        return np.argmax(mat_q[estado_actual.value])
    else:
        print("Acción aleatoria")
        return random.randint(0, 2)
    
def go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    """
    Hace que el robot avance recto.

    Args:
        robot: Robot.
        leftWheel: Rueda izquierda.
        rightWheel: Rueda derecha.
        leftEncoder: Encoder izquierdo.
        rightEncoder: Encoder derecho.    
    """

    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + FORWARD_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED)
        rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    """
    Hace que el robot gire a la izquierda.

    Args:
        robot: Robot.
        leftWheel: Rueda izquierda.
        rightWheel: Rueda derecha.
        leftEncoder: Encoder izquierdo.
        rightEncoder: Encoder derecho.
    """

    initial_position = rightEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and rightEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(-4)
        rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    """
    Hace que el robot gire a la derecha.

    Args:
        robot: Robot.
        leftWheel: Rueda izquierda.
        rightWheel: Rueda derecha.
        leftEncoder: Encoder izquierdo.
        rightEncoder: Encoder derecho.
    """

    initial_position = leftEncoder.getValue()

    while robot.step(TIME_STEP) != -1 and leftEncoder.getValue() < initial_position + TURN_DELTA:
        leftWheel.setVelocity(CRUISE_SPEED)
        rightWheel.setVelocity(-4)

def perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder):
    """
    Realiza la acción indicada.

    Args:
        action: Acción a realizar.
        Accion: Enumerado con las acciones.
        robot: Robot.
        leftWheel: Rueda izquierda.
        rightWheel: Rueda derecha.
        leftEncoder: Encoder izquierdo.
        rightEncoder: Encoder derecho.
    """

    # Si la acción es 0, girar a la derecha.
    if action == Accion.A1.value:
        turn_right(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 1, girar a la izquierda.
    elif action == Accion.A2.value:
        turn_left(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    # Si la acción es 2, ir recto.
    elif action == Accion.A3.value:
        go_straight(robot, leftWheel, rightWheel, leftEncoder, rightEncoder)

# Funciones de inicialización. ####################################################################

def init():
    """
    Inicializa el robot y obtiene los sensores y motores.

    Returns:
        robot: Robot.
        ir_sensors: Sensores infrarrojos.
        leftWheel: Rueda izquierda.
        rightWheel: Rueda derecha.
        leftEncoder: Encoder izquierdo.
        rightEncoder: Encoder derecho.
        learning_rate: Tasa de aprendizaje.
        gamma_value: Factor de descuento.
        mat_q: Matriz Q.
        visitas: Matriz de visitas.
        estado_actual: Estado actual.
        Estado: Enumerado con los estados.
        Accion: Enumerado con las acciones.
    """

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
    mat_q = np.zeros((3,3))
    visitas = np.zeros((3,3))

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
           gamma_value, mat_q, visitas, \
           estado_actual, Estado, Accion

# Funciones de debug y visualización. #############################################################

def print_q_matrix(mat_q):
    """
    Función que imprime la matriz Q.

    Args:
        mat_q: Matriz Q.
    """

    print("Matriz Q:")
    print("      A1     A2     A3")
    print("-----------------------")

    # Mostrar la matriz Q con redondeo a 2 decimales
    for i in range(mat_q.shape[0]):
        row = f"S{i+1}: "
        for j in range(mat_q.shape[1]):
            row += f"{np.round(mat_q[i][j], 2):>5} "
        print(row)
    print("-----------------------")

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
    gamma_value, mat_q, visitas, \
    estado_previo, Estado, Accion = init()

    cnt = 0 # Contador de iteraciones
    sensor_values = check_sensors(ir_sensors) # Obtener los valores de los sensores infrarrojos

    # Bucle principal de la simulación.
    while robot.step(TIME_STEP) != -1:
        print("Iteración número: ", cnt)

        # Realizar acción
        action = pick_action(estado_previo, mat_q, cnt)
        perform_action(action, Accion, robot, leftWheel, rightWheel, leftEncoder, rightEncoder)
    
        # Actualizar matriz Q
        new_sensor_values = (check_sensors(ir_sensors))
        nuevo_estado = check_estado(new_sensor_values, Estado)
        refuerzo = check_refuerzo(sensor_values, new_sensor_values)
        actualizar_matriz_q(refuerzo, action, estado_previo, nuevo_estado, learning_rate, gamma_value, visitas, mat_q)
        print_q_matrix(mat_q)

        # Esquivar obstáculos
        while (ir_sensors[2].getValue() > IR_THRESHOLD or \
               ir_sensors[3].getValue() > IR_THRESHOLD or \
               ir_sensors[4].getValue() > IR_THRESHOLD) and \
               robot.step(TIME_STEP) != -1:
                speed_offset = 0.2 * (CRUISE_SPEED - 0.03 * ir_sensors[3].getValue());
                speed_delta = 0.03 * ir_sensors[2].getValue() - 0.03 * ir_sensors[4].getValue()
                leftWheel.setVelocity(speed_offset + speed_delta)
                rightWheel.setVelocity(speed_offset - speed_delta)

        # Actualizar variables
        estado_previo = nuevo_estado
        sensor_values = new_sensor_values
        cnt += 1

        print("###########################\n\n")

if __name__ == "__main__":
    main()
