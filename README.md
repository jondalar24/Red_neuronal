# Simulación paso a paso de una neurona artificial con Python

Este repositorio contiene un ejemplo ilustrativo de cómo funciona una **neurona artificial** a nivel básico, implementado en Python con `numpy` y `matplotlib`. El objetivo es mostrar de forma didáctica los conceptos de **propagación hacia adelante (forward propagation)**, **cálculo de error**, **retropropagación (backpropagation)** y **actualización de pesos y bias**.

## 🔢 Objetivo
Simular cómo una red neuronal con:
- 2 entradas (`x1`, `x2`)
- 1 neurona
- pesos y bias inicializados manualmente

puede aprender a ajustar sus parámetros mediante descenso de gradiente para aproximarse a un valor objetivo (`y_true`).

## 📊 Etapas del código

1. **Inicialización**:
   - Entradas `x = [0.5, 0.85]`
   - Pesos `w = [0.15, 0.2]`
   - Bias `b = 0.4`
   - Valor objetivo `y_true = 1`

2. **Forward propagation**:
   - Se calcula `z = w·x + b`
   - Se aplica la función sigmoide: `a = sigmoid(z)`

3. **Cálculo del error**:
   - Se usa el error cuadrático medio (MSE): `error = 0.5 * (y_true - a)^2`

4. **Backpropagation**:
   - Se calcula el gradiente del error con respecto a cada peso (`dw`) y al bias (`db`)

5. **Actualización de parámetros**:
   - Se aplica descenso de gradiente: `w = w - η * dw` y `b = b - η * db`

6. **Visualización del paso a paso**:
   - Se construye un `DataFrame` para mostrar todos los pasos y valores.

## 📒 Tecnologías utilizadas
- Python 3
- Numpy
- Pandas
- Matplotlib (opcional para visualizar más adelante)

## 🔍 Ejemplo de salida esperada
Una tabla que muestra:
- El valor inicial de `z`, `a`, y el `error`
- Los gradientes calculados
- La actualización final de `w` y `b`

## 🌟 Motivación
Este ejercicio es ideal para quien está empezando con redes neuronales y quiere comprender cómo funciona el **entrenamiento de una red** desde cero. Entender este paso es esencial antes de saltar a frameworks como TensorFlow o PyTorch.

---

**Autor**: [Tu nombre o usuario de GitHub]

**Curso**: IBM AI Engineering - Coursera

**Tema**: Entrenamiento de redes neuronales (Módulo 2 - Forward y Backpropagation)

---

Si te ha resultado útil, no dudes en darle una estrella al repositorio o contactarme para más ejemplos.

