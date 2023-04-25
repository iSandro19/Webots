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

# Time step por defecto para el controlador.
TIME_STEP = 16

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

###################################################################################################
# FUNCIONES AUXILIARES
###################################################################################################

# Funciones de comprobación de los sensores, estados y refuerzo. ##################################

def check_sensors(ir_sensors):
    return [ir_sensors[8].getValue(), ir_sensors[9].getValue(),
            ir_sensors[10].getValue(), ir_sensors[11].getValue()]
    
def check_estado(sensor_values, Estado):
    if sensor_values[1] > 750 and sensor_values[3] < 500:
        return Estado.S1
    elif sensor_values[2] > 750 and sensor_values[0] < 500:
        return Estado.S2
    return Estado.S3
    
def check_refuerzo(new_sensor_values, prev_sensor_values):
    if all(value < 500 for value in prev_sensor_values) and \
       all(value < 500 for value in new_sensor_values):
        return 1
    elif all(value < 500 for value in prev_sensor_values) and \
         not all(value < 500 for value in new_sensor_values):
        return -1
    elif all(value > 750 for value in prev_sensor_values) and \
         all(value > 750 for value in new_sensor_values):
        return -1
    elif all(value > 750 for value in prev_sensor_values) and \
         not all(value > 750 for value in new_sensor_values):
        return 1
    elif sum(i > 750 for i in prev_sensor_values) < \
         sum(i > 750 for i in new_sensor_values):
        return -1
    else:
        return 1
    
# Funciones de actualización. #####################################################################

def actualizar_refuerzo(refuerzo, action, prev_estado, nuevo_estado, learning_rat, visitas):
    visitas[prev_estado.value][action] += 1
    learning_rate = 1 / (1 + visitas[prev_estado.value][action])
    #mat_q[prev_estado.value][action] = (1-learning_rate) * mat_q[prev_estado.value][action] + learning_rate * (refuerzo + gamma_value * np.argmax(mat_q[nuevo_estado.value]))
    mat_q = np.identity(3)
    return learning_rate

# Funciones de ejecución de acciones. #############################################################

def pick_random_action():
    return random.randint(0, 2)
    
def pick_action(estado_actual, mat_q):
    return np.argmax(mat_q[estado_actual.value])

def go_straight(Accion, leftWheel, rightWheel):
    accion_actual = Accion.A1
    leftWheel.setVelocity(CRUISE_SPEED)
    rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_left(Accion, leftWheel, rightWheel):
    accion_actual = Accion.A2
    leftWheel.setVelocity(-CRUISE_SPEED)
    rightWheel.setVelocity(CRUISE_SPEED)
    
def turn_right(Accion, leftWheel, rightWheel):
    accion_actual = Accion.A3
    leftWheel.setVelocity(CRUISE_SPEED)
    rightWheel.setVelocity(-CRUISE_SPEED)

def perform_action(action, Accion, leftWheel, rightWheel):
    if action == 0:
        turn_right(Accion, leftWheel, rightWheel)
    elif action == 1:
        turn_left(Accion, leftWheel, rightWheel)
    elif action == 2:
        go_straight(Accion, leftWheel, rightWheel)

###################################################################################################
# MAIN
###################################################################################################
def main():
    """
    Función principal de la práctica.
    """

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

    leds = [robot.getDevice("front left led"), robot.getDevice("front right led"), robot.getDevice("rear led")]

    leftWheel = robot.getDevice("left wheel motor")
    rightWheel = robot.getDevice("right wheel motor")
    leftWheel.getPositionSensor().enable(TIME_STEP)
    rightWheel.getPositionSensor().enable(TIME_STEP)
    leftWheel.setPosition(float('inf'))
    rightWheel.setPosition(float('inf'))
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

    last_display_second = 0

    learning_rate = 0.5
    gamma_value = 0.5

    mat_q = np.identity(3)
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
    accion_actual = Accion.A1

    while robot.step(TIME_STEP) != -1:
        display_second = robot.getTime()
        if display_second != last_display_second:
            last_display_second = display_second
            
            # Comprobación se seguridad para las paredes
            if (ir_sensors[2].getValue() > 300 or ir_sensors[3].getValue() > 300 or ir_sensors[4].getValue() > 300):
                speed_offset = 0.2 * (CRUISE_SPEED - 0.03 * ir_sensors[3].getValue());
                speed_delta = 0.03 * ir_sensors[2].getValue() - 0.03 * ir_sensors[4].getValue()
                leftWheel.setVelocity(speed_offset + speed_delta)
                rightWheel.setVelocity(speed_offset - speed_delta)
            
            # Q-Learning
            else:
                sensor_values = check_sensors(ir_sensors)
                sensors_hist.append(sensor_values)
                estado_actual = check_estado(sensor_values, Estado)
                
                action = pick_action(estado_actual, mat_q)
                print(mat_q)
                perform_action(action, Accion, leftWheel, rightWheel)
            
                sensors_hist.append(check_sensors(ir_sensors))
                new_sensor_values = sensors_hist[len(sensors_hist)-3]
                nuevo_estado = check_estado(new_sensor_values, Estado)
                refuerzo = check_refuerzo(sensor_values, new_sensor_values)
                learning_rate = actualizar_refuerzo(refuerzo, action, estado_actual, nuevo_estado, learning_rate, visitas)

if __name__ == "__main__":
    main()
