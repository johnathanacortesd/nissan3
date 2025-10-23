# 🚀 Procesador Inteligente de Dossiers Nissan v4.0

Herramienta automatizada para procesar, limpiar y analizar dossiers de noticias usando Machine Learning e Inteligencia Artificial.

## 📋 ¿Qué hace esta aplicación?

Este procesador toma archivos Excel con noticias y los enriquece automáticamente con:

- **Análisis de sentimiento** (Positivo/Neutro/Negativo)
- **Clasificación automática de temas**
- **Detección de duplicados**
- **Normalización y limpieza de datos**
- **Reorganización inteligente de columnas**

## ✨ Características principales

### 🤖 Inteligencia Artificial
- **Análisis de Tono**: Clasifica automáticamente el sentimiento de cada noticia
- **Clasificación Temática**: Asigna temas relevantes usando modelos de ML
- **Detección de Duplicados**: Identifica noticias repetidas de forma optimizada

### 🧹 Limpieza de Datos
- Normalización de tipos de medios (Prensa, Radio, TV, Internet)
- Mapeo automático de regiones según el medio
- Corrección de formatos de fecha y hora
- Limpieza de títulos y resúmenes

### 🔗 Gestión de Enlaces
- Extracción de hipervínculos desde celdas Excel
- Reorganización inteligente de links según tipo de medio
- Separación de links de notas y streaming

### 📊 Expansión de Datos
- Expande filas automáticamente cuando hay múltiples menciones de empresa
- Mantiene la integridad de los datos originales

## 🛠️ Requisitos

### Archivos necesarios:
1. **Dossier principal** (Excel): Archivo con las noticias a procesar
2. **Configuracion.xlsx**: Archivo con mapeos y configuraciones
3. **pipeline_sentimiento.pkl**: Modelo ML para análisis de sentimiento
4. **pipeline_tema.pkl**: Modelo ML para clasificación de temas

### Bibliotecas Python:
```
streamlit
pandas
openpyxl
joblib
numpy
nltk
scikit-learn
```

## 🧠 Modelos de Machine Learning

### ¿Qué son los archivos .pkl?

Los archivos `.pkl` (pickle) son modelos de Machine Learning **pre-entrenados y guardados** que contienen:

- **El algoritmo completo** listo para usar
- **Los parámetros aprendidos** durante el entrenamiento
- **El pipeline de preprocesamiento** (limpieza, vectorización, etc.)

**Ventaja**: No necesitas entrenar el modelo cada vez, solo cargarlo y usarlo instantáneamente.

### 📦 pipeline_sentimiento.pkl

**Función**: Analiza el sentimiento/tono de las noticias

**Modelo usado**: **LinearSVC** (Support Vector Classifier lineal)
- Algoritmo de clasificación supervisado muy eficiente
- Optimizado con GridSearchCV para encontrar mejores parámetros
- Parámetros ajustados: C (regularización), max_features, ngram_range

**Entrada**: Título + Resumen de la noticia  
**Salida**: Clasificación en 3 categorías:
- `Positivo` (1)
- `Neutro` (0)
- `Negativo` (-1)

**Cómo funciona**:
1. Toma el texto combinado (título + resumen)
2. **TF-IDF Vectorizer**: Convierte texto en vectores numéricos ponderados
3. **LinearSVC**: Clasifica usando márgenes de separación óptimos
4. Devuelve la etiqueta de sentimiento

**Precisión esperada**: ~85-95% según calidad de datos de entrenamiento

### 🏷️ pipeline_tema.pkl

**Función**: Clasifica automáticamente el tema de cada noticia

**Modelo usado**: **LinearSVC** (Support Vector Classifier lineal)
- Entrenado en modo multiclase (múltiples categorías temáticas)
- Filtra automáticamente clases con menos de 4 muestras
- Optimizado con GridSearchCV (60K-70K features)

**Entrada**: Texto preprocesado (sin stopwords en español, tokenizado)  
**Salida**: Categoría temática (ej: "Ventas", "Producto", "Servicio Técnico", etc.)

**Cómo funciona**:
1. **Preprocesamiento**: Elimina stopwords (palabras vacías en español)
2. **TF-IDF Vectorization**: ngrams (1,1) o (1,2) con min_df=2
3. **LinearSVC**: Clasifica según patrones aprendidos con C=0.8-1.0
4. Asigna la categoría más probable

**Precisión esperada**: Variable según número de categorías (típicamente 70-90%)

### 🔄 Pipeline completo

Cada `.pkl` contiene un **pipeline de scikit-learn** optimizado:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),        # Unigramas y bigramas
        min_df=2,                  # Mínimo 2 documentos
        max_features=65000,        # Máximo de características
        token_pattern=r'\b\w+\b'   # Patrón de tokenización
    )),
    ('clf', LinearSVC(
        C=1.0,                     # Regularización
        random_state=42,           # Reproducibilidad
        dual="auto"                # Optimización automática
    ))
])
```

**Ventajas de este pipeline**:
- ✅ Procesa texto crudo directamente
- ✅ No requiere pasos intermedios manuales
- ✅ Optimizado con GridSearchCV (3-fold cross-validation)
- ✅ Reproducible (random_state fijo)
- ✅ Versiones estables (NumPy 1.26.4, scikit-learn 1.3.2)

## 🎓 Entrenamiento de Modelos

Si necesitas **re-entrenar** los modelos con tus propios datos:

### Requisitos para entrenar:
- Google Colab (recomendado) o entorno Python local
- Archivo Excel con columnas: `resumen`, `tono`, `tema`
- Mínimo 4 muestras por clase (el script filtra automáticamente)

### Proceso de entrenamiento:

1. **Ejecuta la Celda 1**: Instala dependencias estables
   - NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.3.2
   - El entorno se reiniciará automáticamente

2. **Ejecuta la Celda 2**: Entrena los modelos
   - Sube tu archivo Excel de entrenamiento
   - GridSearchCV optimiza hiperparámetros automáticamente
   - Tiempo estimado: 5-30 minutos según tamaño de datos

3. **Descarga**: Obtienes `modelos_optimizados.zip`
   - Contiene: `pipeline_sentimiento.pkl` y `pipeline_tema.pkl`
   - Listos para usar en la aplicación

### Datos de entrenamiento:

```
| resumen                          | tono      | tema            |
|----------------------------------|-----------|-----------------|
| "Excelente servicio postventa"   | Positivo  | Servicio        |
| "Presentan nuevo modelo SUV"     | Neutro    | Producto        |
| "Retiro de vehículos por fallas" | Negativo  | Servicio Técnico|
```

**Notas importantes**:
- El modelo de tema filtra clases con < 4 muestras automáticamente
- Se usa validación cruzada (3-fold) para evitar overfitting
- Los mejores parámetros se seleccionan automáticamente

## 📁 Estructura de `Configuracion.xlsx`

El archivo debe contener 4 hojas con la siguiente estructura:

### Hoja `Regiones`
| Medio | Región |
|-------|--------|
| nombre_medio | Región asignada |

### Hoja `Internet`
| Medio Original | Medio Mapeado |
|----------------|---------------|
| nombre_web | Nombre normalizado |

### Hoja `Menciones`
| Mención Original | Mención Mapeada |
|------------------|-----------------|
| mencion_cruda | Mención estándar |

### Hoja `Mapa_Temas`
| Temas Generales - Tema | Tema |
|------------------------|------|
| tema_detallado | Categoría final |

## 🚀 Cómo usar

1. **Inicia la aplicación**:
   ```bash
   streamlit run app.py
   ```

2. **Sube tus archivos**:
   - Arrastra el archivo Dossier (Excel)
   - Arrastra el archivo Configuracion.xlsx

3. **Procesa**:
   - Haz clic en "▶️ Iniciar Proceso Completo"
   - Espera mientras se procesan los datos

4. **Descarga resultados**:
   - Descarga el archivo Excel procesado
   - Revisa la previsualización en pantalla

## 📊 Proceso paso a paso

El procesador ejecuta 8 pasos:

1. **Carga de modelos ML** y configuración
2. **Lectura y expansión** del dossier por menciones
3. **Limpieza y normalización** de datos
4. **Reorganización de columnas** según tipo de medio
5. **Detección de duplicados** optimizada
6. **Análisis con IA** (tono y tema)
7. **Homogeneización de temas** por título
8. **Generación del archivo final**

## 📈 Métricas mostradas

Al finalizar, verás:
- **Filas totales procesadas**
- **Filas marcadas como duplicadas**
- **Filas únicas analizadas**

## 🔧 Funciones principales

### `load_ml_models()`
Carga los modelos de ML pre-entrenados para sentimiento y temas.

### `read_and_expand_dossier()`
Lee el Excel, extrae hipervínculos y expande filas por mención de empresa.

### `run_full_process()`
Ejecuta el flujo completo de procesamiento con barra de progreso.

## ⚠️ Consideraciones

- Las fechas deben estar en formato válido (DD/MM/YYYY preferiblemente)
- El archivo debe tener las columnas esperadas por el sistema
- Los modelos ML deben estar en la misma carpeta que `app.py`
- Se requiere conexión a internet la primera vez (descarga recursos NLTK)

## 📝 Columnas del archivo final

El archivo procesado incluye:
- ID Noticia, Fecha, Hora
- Medio, Tipo de Medio, Región
- Título, Autor/Conductor
- Dimensión, Duración, CPE, Tier, Audiencia
- **Tono** (generado por IA)
- **Tema** (generado por IA)
- Resumen, Links, Menciones

## 🎯 Casos de uso

- Análisis de cobertura mediática
- Monitoreo de reputación de marca
- Generación de reportes automatizados
- Clasificación masiva de noticias
- Detección de tendencias en medios

## 🤝 Contribuciones

Este es un proyecto interno creado por Johnathan Cortés, optimizado para procesamiento de dossiers.

---

**Versión**: 4.0  
**Última actualización**: Agosto de 2025  
**Optimizado para**: Alto volumen de datos
