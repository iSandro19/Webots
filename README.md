# Prácticas con Webots

## Robotica Q8 2022-2023

Este repositorio contiene dos prácticas relacionadas con el control de robots autónomos en un entorno simulado. Estas prácticas fueron desarrolladas como parte de las práctica de la asignatura de Robótica del 4o curso del Grado de Ingeniería Informática en la FIC (UDC) y están escritas en Python.

## Práctica 1: Estado de Posicionamiento, Mapeo, Mapeo y Patrullaje

La primera práctica se centra en el estado de posicionamiento y mapeo del robot. El objetivo principal es permitir que el robot se posicione correctamente en el entorno y realice un mapeo del mismo. El código implementa un algoritmo que utiliza sensores infrarrojos para detectar obstáculos y tomar decisiones de movimiento. Una vez que el robot ha completado el proceso de posicionamiento, comienza a explorar y patrullar el entorno, al mismo tiempo que continúa actualizando el mapa del área. Utilizando los datos de los sensores infrarrojos, el robot toma decisiones de movimiento para evitar obstáculos y seguir patrullando el área de manera eficiente. Si se detecta un objetivo específico con la cámara (un objeto amarillo en este caso), el robot pasa al estado de vuelta a la posición inicial y parada. Los mapas de demostración se pueden encontrar en la carpeta `worlds` del repositorio.

# Práctica 2: Q-Learning

El objetivo de la práctica es entrenar a un robot para que siga una línea negra en un entorno simulado utilizando sensores infrarrojos y actuadores de ruedas. El algoritmo utiliza una matriz Q para almacenar los valores de acción-estado y utiliza el refuerzo recibido para actualizar estos valores durante el entrenamiento. Se utiliza un bucle principal en el que se leen los sensores, se determina el estado actual, se elige una acción y se realiza esa acción en el entorno simulado. A continuación, se actualiza la matriz Q en función del refuerzo recibido y se repite el proceso. El código también incluye funciones auxiliares para comprobar los sensores, determinar el estado y el refuerzo, así como funciones de inicialización y visualización. Todo esto se realiza para que el Khepera IV aprenda a circular por una linea negra sobre un fondo blanco. Los mapas de demostración se pueden encontrar en la carpeta `worlds` del repositorio.

## Instrucciones de Uso

1. Clona este repositorio en tu máquina local.
2. Copia las carpetas de controllers y worlds en el directorio de instalación de Webots ("../Webots/resources/projects/").
3. Abre Webots y selecciona el mundo que deseas ejecutar.
4. Selecciona el controlador que deseas ejecutar.
5. Ejecuta el mundo.

""" Nota: si el rendimiento no es el adecuado puede aumentar el time step en el mundo (definido a 16) preferiblemente con múltiplos de ese número. """

## Requisitos del Sistema

- Webots
- Python
- Numpy

## Autores

- Óscar Alejandro Manteiga Seoane
- Antonio Vila Leis
