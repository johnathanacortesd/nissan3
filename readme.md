# 🚀 Procesador Inteligente de Dossiers Nissan v4.0

Herramienta automatizada para procesar, limpiar y analizar dossiers de noticias usando Machine Learning e Inteligencia Artificial.

## 📋 ¿Qué hace esta aplicación?

Este procesador toma archivos Excel con noticias y menciones de prensa, y los enriquece automáticamente con:

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
```

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

Este es un proyecto interno optimizado para procesamiento de alto volumen de dossiers de prensa.

## 📄 Creado por Johnathan Cortés


**Versión**: 4.0  
**Última actualización**: 2024  
**Optimizado para**: Alto volumen de datos
