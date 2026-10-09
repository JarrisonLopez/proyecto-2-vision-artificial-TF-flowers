# Resultados

Salidas del notebook `proyecto_2_vision_artificial.ipynb`.

- `resumen_experimentos.csv`: métricas de validación de las ocho configuraciones.
- `historiales.json`: curvas de entrenamiento y validación.
- `mejor_modelo.json`: configuración seleccionada por macro-F1 de validación.
- `evaluacion_prueba.json`: métricas del conjunto de prueba.
- `conclusiones.txt`: lectura de esas métricas.
- `figuras/`: gráficos del análisis, del entrenamiento y de la evaluación.

Los pesos `.keras`, la carpeta `checkpoints/` y `resultados.zip` permanecen en el equipo local y no forman parte del repositorio. El dataset `tf_flowers` tampoco se guarda aquí: el notebook lo obtiene con TensorFlow Datasets.
