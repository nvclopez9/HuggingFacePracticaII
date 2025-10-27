# 🏉 Proyecto Hugging Face – “Skate IA”

**Curso:** Curso de Especialización en Inteligencia Artificial y Big Data (2025–2026)
**Módulo:** Modelos de IA
**Centro:** CFIP César Manrique

---

## 🧠 Descripción del proyecto

Este proyecto forma parte de la **Práctica II del módulo Modelos de IA**.
Su objetivo es demostrar la interacción entre diferentes modelos de **Inteligencia Artificial** usando la plataforma **Hugging Face** y la librería de interfaz **Gradio**.

La idea surge de una simulación en la que el **Ayuntamiento de Monzalbarba** desea modernizarse y explorar las posibilidades de la IA.
Para ello, desarrollamos una mini aplicación que permite **diseñar y personalizar un skate** mediante el uso de modelos de generación, transformación e interpretación de imágenes.

---

## ⚙️ Funcionalidad general

La aplicación sigue un flujo de **cuatro tareas de IA conectadas (tasks)**, donde la salida de cada modelo se convierte en la entrada del siguiente:

```
text → image → transformed image → caption → emoji
```

### 🧉 Flujo completo:

1. **Text-to-Image (`nitrosocke/Arcane-Diffusion`)**
   Genera una imagen artística a partir de un texto descriptivo.
2. **Image-to-Image (`runwayml/stable-diffusion-v1-5`)**
   Transforma la imagen generada para simular su aplicación en la parte inferior de un skate.
3. **Image-to-Text (`Salesforce/blip-image-captioning-base`)**
   Describe automáticamente la imagen final generada.
4. **Text-to-Emoji (`KomeijiForce/t5-base-emojilm`)**
   Convierte el texto del usuario en una secuencia de emojis representativos.

---

## 🚀 Ejemplo de uso

**Entrada (prompt):**

```
un dragón de fuego estilo Arcane
```

**Salida esperada:**

1. Imagen generada con un dragón rojo en estilo Arcane.
2. Imagen transformada aplicada sobre un skate.
3. Descripción generada: “a red dragon printed on a skateboard”.
4. Emojis generados: 🔥🐉🎨

---

## 💻 Instalación y ejecución

### 1⃣ Clonar el repositorio

```bash
git clone https://github.com/nvclopez9/HuggingFacePracticaII.git
cd HuggingFacePracticaII
```

### 2⃣ Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate   # En Linux/Mac
venv\Scripts\activate      # En Windows
```

### 3⃣ Instalar dependencias necesarias

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate gradio pillow
```

### 4⃣ Ejecutar la aplicación

```bash
python app.py
```

> 💡 **Nota:** El script principal del proyecto es el que contiene el código mostrado en la práctica (el que incluye la interfaz Gradio).
> Se recomienda disponer de una **GPU NVIDIA** para un rendimiento óptimo, aunque el código también puede ejecutarse en CPU (más lento).

---

## 🧩 Estructura del código

```
├── app.py                     # Código principal con los modelos y la interfaz Gradio
├── requirements.txt          # (Opcional) Librerías necesarias
├── README.md                 # Documento de descripción del proyecto
└── outputs/                  # Carpeta sugerida para guardar imágenes generadas
```

---

## 🧠 Modelos utilizados

| Task               | Modelo                                  | Descripción                                          |
| ------------------ | --------------------------------------- | ---------------------------------------------------- |
| **Text-to-Image**  | `nitrosocke/Arcane-Diffusion`           | Genera imágenes con estilo artístico tipo *Arcane*.  |
| **Image-to-Image** | `runwayml/stable-diffusion-v1-5`        | Modifica imágenes manteniendo coherencia visual.     |
| **Image-to-Text**  | `Salesforce/blip-image-captioning-base` | Genera descripciones textuales a partir de imágenes. |
| **Text-to-Emoji**  | `KomeijiForce/t5-base-emojilm`          | Convierte texto en emojis representativos.           |

---

## 🗾 Resultados y tiempos promedio

| Ejecución | Tarea          | Modelo               | Tiempo aprox. | Comentario                            |
| --------- | -------------- | -------------------- | ------------- | ------------------------------------- |
| 1         | Text-to-Image  | Arcane-Diffusion     | 100s          | Imagen coherente, buen estilo visual. |
| 2         | Image-to-Image | Stable Diffusion 1.5 | 76s           | Aplicación al skate correcta.         |
| 3         | Image-to-Text  | BLIP Captioning      | 1s            | Descripción precisa y natural.        |
| 4         | Text-to-Emoji  | T5 Emojilm           | 0.5s          | Emojis coherentes y divertidos.       |

---

## 🎨 Interfaz Gradio

La interfaz fue desarrollada con **Gradio Blocks**, utilizando botones y campos organizados por secciones:

* **Textbox:** para ingresar la descripción del diseño.
* **Botones:** generar imagen, regenerar, aplicar al skate, describir imagen y generar emojis.
* **Imágenes:** para mostrar los resultados generados.
* **Textboxes:** para mostrar descripciones y emojis resultantes.

Ejemplo visual del flujo de la interfaz:

```
[Texto de usuario] → [Imagen generada] → [Imagen transformada] → [Descripción] → [Emojis]
```

---

## 💬 Comentarios finales

El proyecto demuestra cómo es posible **integrar múltiples tareas de IA** con recursos open source, sin depender de plataformas en la nube.
El uso combinado de **Hugging Face**, **Transformers**, **Diffusers** y **Gradio** facilita la creación de demos funcionales de IA que pueden aplicarse tanto a proyectos educativos como a usos creativos.

> “Nos sorprendió la calidad visual del modelo Arcane-Diffusion y la rapidez de BLIP y T5.
> Aunque los tiempos de los modelos Diffusion son largos, el resultado final demuestra el potencial de estas herramientas.”

---

## 💎 Enlace del repositorio

🔗 [https://github.com/nvclopez9/HuggingFacePracticaII.git](https://github.com/nvclopez9/HuggingFacePracticaII.git)

---

## 🧩 Créditos

**Centro:** CFIP César Manrique
**Año académico:** 2025–2026
**Asignatura:** Modelos de IA
