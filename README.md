# Clasificación de flores con transferencia de aprendizaje

Proyecto 2 de Visión artificial. Ingeniería de Sistemas, Universidad de Medellín.

Clasificación de fotografías de flores en cinco clases mediante transferencia de aprendizaje y fine-tuning de VGG16 y ResNet50.

## Integrantes

- Jarrison Andrés López Roldán
- Juliana Andrea Betancur Blandón

## Objetivo

Comparar VGG16 y ResNet50, ambas inicializadas con pesos de ImageNet, sobre una malla de hiperparámetros. El modelo se selecciona con el macro-F1 de validación. El conjunto de prueba se reserva para la evaluación final.

## Dataset

- Nombre: [`tf_flowers`](https://www.tensorflow.org/datasets/catalog/tf_flowers)
- Origen: TensorFlow Team (2019), paquete `flower_photos`
- 3.670 imágenes RGB y un único split oficial (`train`)
- Tarea: clasificación supervisada multiclase

| Clase | Imágenes | Porcentaje |
| --- | ---: | ---: |
| dandelion | 898 | 24.47 % |
| tulips | 799 | 21.77 % |
| sunflowers | 699 | 19.05 % |
| roses | 641 | 17.47 % |
| daisy | 633 | 17.25 % |

El orden oficial de las etiquetas es `dandelion`, `daisy`, `tulips`, `sunflowers`, `roses`. El notebook descarga el dataset con `tfds.load("tf_flowers")`.

## Metodología

1. Carga con TensorFlow Datasets.
2. Análisis de frecuencias, dimensiones y ejemplos de las cinco clases.
3. Partición estratificada 70 % / 15 % / 15 % (2.568, 551 y 551 imágenes). La prueba no interviene en el entrenamiento ni en la selección.
4. Redimensionado a 224×224 con `resize_with_pad`, conservando la relación de aspecto.
5. Preprocesamiento propio de cada arquitectura: `vgg16.preprocess_input` y `resnet50.preprocess_input`.
6. Aumento solo en entrenamiento: volteo horizontal, rotación leve, brillo y contraste.
7. Cabeza clasificadora: `GlobalAveragePooling2D`, Dropout y softmax de cinco clases.
8. Dos fases. Primero se entrena la cabeza con la base congelada y Adam a `1e-3`. Después se descongelan las últimas capas convolucionales y se ajustan con Adam a `1e-5`.
9. En ResNet50, BatchNormalization permanece congelada y la base se ejecuta en modo inferencia.
10. EarlyStopping y ModelCheckpoint siguen el macro-F1 de validación.

En las redes implementadas, VGG16 tiene 14.717.253 parámetros totales y ResNet50 tiene 23.597.957.

## Arquitecturas e hiperparámetros

Se compararon las dos arquitecturas, los tamaños de lote 16 y 32, y los valores de dropout 0.3 y 0.5. Son ocho experimentos. Las tasas de aprendizaje permanecen fijas: `1e-3` en la cabeza y `1e-5` en el fine-tuning.

| Arquitectura | Lote | Dropout | Macro-F1 validación | Accuracy validación | Tiempo (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 16 | 0.5 | 0.9468 | 0.9474 | 7.7 |
| ResNet50 | 32 | 0.5 | 0.9455 | 0.9456 | 7.7 |
| ResNet50 | 16 | 0.3 | 0.9435 | 0.9437 | 8.9 |
| ResNet50 | 32 | 0.3 | 0.9432 | 0.9437 | 9.1 |
| VGG16 | 16 | 0.5 | 0.9202 | 0.9220 | 8.4 |
| VGG16 | 16 | 0.3 | 0.9140 | 0.9147 | 8.5 |
| VGG16 | 32 | 0.3 | 0.9043 | 0.9056 | 8.6 |
| VGG16 | 32 | 0.5 | 0.9012 | 0.9020 | 8.6 |

La tabla está ordenada por el criterio de selección: mayor macro-F1 de validación. La accuracy de validación y el tiempo solo desempatan. Las métricas de prueba no participan en esa elección.

## Modelo seleccionado

ResNet50, lote 16 y dropout 0.5 (`resnet50_lote16_dropout0.50`).

| Conjunto | Accuracy | Precisión macro | Recall macro | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Validación | 0.9474 | — | — | 0.9468 |
| Prueba (551 imágenes) | 0.9365 | 0.9348 | 0.9354 | 0.9351 |

El tiempo acumulado de los ocho experimentos fue de 67.4 minutos. El fine-tuning elevó el macro-F1 de validación del modelo elegido en 0.0358.

F1 de prueba por clase: dandelion 0.9665; daisy 0.9524; sunflowers 0.9333; tulips 0.9205; roses 0.9026. La confusión más frecuente es entre tulips y roses.

## Conclusiones

ResNet50 superó a VGG16 en las cuatro combinaciones de lote y dropout. El mejor macro-F1 de validación fue 0.9468 y el de prueba, 0.9351. La diferencia, -0.0118, es una caída pequeña y no demuestra por sí sola un problema grave de generalización.

En la mejor época de fine-tuning, la accuracy de entrenamiento fue 0.9498 y la de validación 0.9474. Esa proximidad no basta para afirmar que no hay sobreajuste.

ResNet50 tiene más parámetros totales que VGG16 en esta implementación. Los tiempos medios, 8.3 y 8.5 minutos, son cercanos, de modo que el número de parámetros no explica por sí solo la duración de las corridas.

El modelo distingue cinco categorías de fotografías centradas en la flor y parte de filtros de ImageNet. No constituye un identificador botánico general.

## Estructura

```text
proyecto_2_vision_artificial.ipynb   Notebook ejecutado
README.md                            Este archivo
requirements.txt                     Dependencias
.gitignore                           Exclusiones del repositorio
resultados/                          Métricas, conclusiones y figuras
```

## Requisitos y ejecución en Google Colab

El notebook se entrega con las salidas de la ejecución completa. Para reproducirla:

1. Active una GPU: Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU.
2. Abra `proyecto_2_vision_artificial.ipynb`.
3. Sitúe el directorio de trabajo en la carpeta del proyecto.
4. Ejecute las celdas en orden. La instalación usa `requirements.txt` y TensorFlow Datasets descarga `tf_flowers` (218.21 MiB según el catálogo).

`MODO_EXPERIMENTO = "completo"` entrena las ocho configuraciones, cada una con cabeza y fine-tuning. Volver a ejecutar esas celdas repite el entrenamiento. Las dependencias están en `requirements.txt`: TensorFlow, TensorFlow Datasets, NumPy, pandas, Matplotlib, seaborn, scikit-learn y Pillow.

## Referencias

- TensorFlow Datasets. *tf_flowers*. https://www.tensorflow.org/datasets/catalog/tf_flowers
- TensorFlow Team. (2019). *Flowers*. http://download.tensorflow.org/example_images/flower_photos.tgz
- Keras. Guía de transferencia y fine-tuning. https://keras.io/guides/transfer_learning/
- Keras Applications. https://keras.io/api/applications/
- Simonyan, K. y Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. https://arxiv.org/abs/1409.1556
- He, K., Zhang, X., Ren, S. y Sun, J. (2016). Deep residual learning for image recognition. https://arxiv.org/abs/1512.03385
- Russakovsky, O. et al. (2015). ImageNet large scale visual recognition challenge. https://doi.org/10.1007/s11263-015-0816-y

Las referencias completas están en la sección 12 del notebook.
