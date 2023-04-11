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
TIME_STEP = 32

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
    leftWheel = robot.getDevice("left wheel motor")
    rightWheel = robot.getDevice("right wheel motor")

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
    posL = robot.getDevice("left wheel sensor")
    posR = robot.getDevice("right wheel sensor")
    posL.enable(timeStep)
    posR.enable(timeStep)

    # TODO: Obtener y activar otros dispositivos necesarios.
    # ...

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

def turn_left(leftWheel, rightWheel, speed, duration):
    """
    Hacer girar al robot hacia la izquierda durante la duración especificada.
    """

    # Convertir la velocidad de cm/s a rad/s para los motores de las ruedas.
    leftWheel.setVelocity(-speed * MAX_SPEED / 100)
    rightWheel.setVelocity(speed * MAX_SPEED / 100)

    # Esperar la duración especificada.
    time.sleep(duration)

    # Detener los motores de las ruedas.
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

def turn_right(leftWheel, rightWheel, speed, duration):
    """
    Hacer girar al robot hacia la derecha durante la duración especificada.
    """

    # Convertir la velocidad de cm/s a rad/s para los motores de las ruedas.
    leftWheel.setVelocity(speed * MAX_SPEED / 100)
    rightWheel.setVelocity(-speed * MAX_SPEED / 100)

    # Esperar la duración especificada.
    time.sleep(duration)

    # Detener los motores de las ruedas.
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

def move_forward(leftWheel, rightWheel, speed, duration):
    # Convertir la velocidad de cm/s a rad/s para los motores de las ruedas.
    leftWheel.setVelocity(speed * MAX_SPEED / 100)
    rightWheel.setVelocity(speed * MAX_SPEED / 100)

    # Esperar la duración especificada.
    time.sleep(duration)

    # Detener los motores de las ruedas.
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

def main():
    # Activamos los dispositivos necesarios y obtenemos referencias a ellos.
    robot, leftWheel, rightWheel, irSensorList, posL, posR, camera = init_devices(TIME_STEP)

    # Ejecutamos una sincronización para tener disponible el primer frame de la cámara.
    robot.step(TIME_STEP)

    # Avanzar recto durante 5 segundos a una velocidad de crucero.
    move_forward(leftWheel, rightWheel, CRUISE_SPEED, 5)

    # Girar a la derecha durante 2 segundos.
    leftWheel.setVelocity(-CRUISE_SPEED * MAX_SPEED / 100)
    rightWheel.setVelocity(CRUISE_SPEED * MAX_SPEED / 100)
    time.sleep(2)

    # Avanzar recto durante otros 5 segundos a una velocidad de crucero.
    move_forward(leftWheel, rightWheel, CRUISE_SPEED, 5)

    # Detener los motores de las ruedas.
    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

if __name__ == "__main__":
    main()