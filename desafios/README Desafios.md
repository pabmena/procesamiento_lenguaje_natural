# Procesamiento de Lenguaje Natural - Desafíos CEIA FIUBA

Este repositorio contiene la resolución de los cuatro desafíos propuestos en la materia de Procesamiento de Lenguaje Natural I (CEIA FIUBA). Cada desafío se implementó en un notebook de Colab y luego se subió al repositorio para consulta.

---

## Estructura del repositorio

* **Desafio1_Pablo_Menardi.ipynb**: Similaridad y clasificación con el dataset 20 Newsgroups.
* **Desafio2_Pablo_Menardi.ipynb**: Entrenamiento de embeddings personalizados con Gensim y análisis de similitud.
* **Desafio3_Pablo_Menardi.ipynb**: Modelo de lenguaje carácter-a-carácter con LSTM y generación de texto.
* **Desafio4_Pablo_Menardi.ipynb**: Bot QA con esquema encoder–decoder LSTM sobre el subset "volunteers" de ConvAI2.

---

## Desafío 1: Similaridad y clasificación Naïve Bayes

**Objetivos**:

1. Vectorizar documentos (TF-IDF) y medir similitud coseno.
2. Entrenar clasificadores Naïve Bayes (Multinomial y Complement).
3. Analizar similitud entre términos transponiendo la matriz término-documento.

**Principales pasos**:

* Carga del dataset 20 Newsgroups (scikit-learn).
* Vectorización con `TfidfVectorizer` (ngrams 1–2, stop words, min\_df).
* Cálculo de similitud coseno entre documentos y evaluación de coherencia temática.
* Grid search sobre hiperparámetros del vectorizador y ComplementNB para maximizar F1-macro.
* Análisis de similitud léxica entre palabras seleccionadas manualmente.

**Resultados**:

* F1-macro base: \~58.5%.
* F1-macro optimizado (TF-IDF + ComplementNB): ≈70.5%.
* Vecindades por similitud coherentes en documentos y términos.

---

## Desafío 2: Custom embeddings con Gensim

**Objetivos**:

* Entrenar un modelo Word2Vec (skip-gram) sobre corpus de letras de canciones.
* Visualizar pérdidas durante el entrenamiento.
* Analizar similitud semántica entre términos.

**Principales pasos**:

* Descarga y carga de archivos `.txt` con letras de diferentes artistas.
* Preprocesamiento básico: tokenización con `text_to_word_sequence`.
* Construcción y entrenamiento de modelo `Word2Vec` (window, min\_count, negative sampling, skip-gram, epochs).
* Inspección de similitudes (`most_similar`) y entrenamiento de callback para mostrar loss.
* Reducción de dimensionalidad (TSNE) y visualización en 2D/3D.

**Resultados**:

* Pérdida de entrenamiento decreciente por época.
* Vecindarios semánticos coherentes (por ejemplo, palabras relacionadas con "love", "darling").
* Visualización interactiva de embeddings.

---

## Desafío 3: Modelo de lenguaje con LSTM por caracteres

**Objetivos**:

* Construir un modelo de lenguaje carácter-a-carácter.
* Tokenizar el corpus completo de un libro (texto de Julio Verne).
* Entrenar un RNN (SimpleRNN/LSTM) y medir perplejidad en validación.
* Generar texto con estrategia greedy y beam search.

**Principales pasos**:

* Descarga y parseo de HTML de Wikipedia/textos.info.
* Preparación de secuencias deslizantes de longitud fija (contexto).
* Construcción de red neuronal con `CategoryEncoding`, `SimpleRNN`, `Dense` y callback de perplejidad.
* Entrenamiento hasta convergencia (perplejidad de \~7.4 → 3.8).
* Generación de texto y demostración de beam search y sampling a distintas temperaturas.

**Resultados**:

* Perplejidad final en validación: \~3.8.
* Texto generado con gramática y estilo consistentes, aunque con repetición a largos.

---

## Desafío 4: LSTM Bot QA (ConvAI2 volunteers)

**Objetivos**:

* Construir un bot de preguntas y respuestas usando encoder–decoder LSTM.
* Procesar subset "volunteers" de ConvAI2.
* Entrenar un seq2seq con embedding entrenable.

**Principales pasos**:

* Descarga del JSON de volunteers.
* Limpieza y extracción de pares pregunta-respuesta (limite 30 tokens).
* Tokenización y padding con `Tokenizer` y `pad_sequences`.
* Construcción de encoder–decoder LSTM (256 u, embedding 128 d entrenable).
* Entrenamiento (15 épocas) con RMSprop: `loss` train ↓6.85→3.49, `val_loss` ↓5.00→3.48.
* Separación de submodelos para inferencia y función `answer()`.

**Resultados**:

* Respuestas coherentes y gramaticales para saludos y preguntas sencillas.
* Modelo de \~8.1 M parámetros.

---

### Enlaces a notebooks

* Desafío 1: [https://github.com/pabmena/procesamiento\_lenguaje\_natural/blob/main/desafios/Desafio1\_Pablo\_Menardi.ipynb](https://github.com/pabmena/procesamiento_lenguaje_natural/blob/main/Desafio1_Pablo_Menardi.ipynb)
* Desafío 2: [https://github.com/pabmena/procesamiento\_lenguaje\_natural/blob/main/desafios/Desafio2\_Pablo\_Menardi.ipynb](https://github.com/pabmena/procesamiento_lenguaje_natural/blob/main/Desafio2_Pablo_Menardi.ipynb)
* Desafío 3: [https://github.com/pabmena/procesamiento\_lenguaje\_natural/blob/main/desafios/Desafio3\_Pablo\_Menardi.ipynb](https://github.com/pabmena/procesamiento_lenguaje_natural/blob/main/Desafio3_Pablo_Menardi.ipynb)
* Desafío 4: [https://github.com/pabmena/procesamiento\_lenguaje\_natural/blob/main/desafios/Desafio4\_Pablo\_Menardi.ipynb](https://github.com/pabmena/procesamiento_lenguaje_natural/blob/main/Desafio4_Pablo_Menardi.ipynb)

---

**Autor:** Pablo Menardi
**Contacto:** [pablomenardi22@gmail.com](mailto:pablomenardi22@gmail.com)
