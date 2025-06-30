import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, NamedStyle
from collections import defaultdict, Counter
from copy import deepcopy
import datetime
import io
import joblib
import re
import nltk

# --- Configuración de la página ---
st.set_page_config(page_title="Procesador de Dossiers Nissan v2.1", layout="wide")

# --- Descarga NLTK stopwords si es necesario ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Descargando recursos de lenguaje por primera vez...")
    nltk.download('stopwords')
    st.success("Recursos listos.")

# ==============================================================================
# SECCIÓN DE FUNCIONES AUXILIARES
# ==============================================================================
def norm_key(text):
    return re.sub(r'\W+', '', str(text).lower().strip()) if text else ""

def convert_html_entities(text):
    if not isinstance(text, str): return text
    html_entities = {'á': 'á', 'é': 'é', 'í': 'í', 'ó': 'ó', 'ú': 'ú', 'ñ': 'ñ', 'Á': 'Á', 'É': 'É', 'Í': 'Í', 'Ó': 'Ó', 'Ú': 'Ú', 'Ñ': 'Ñ', '\"': '\"', '“': '\"', '”': '\"', '‘': "'", '’': "'", 'Â': '', 'â': '', '€': '', '™': ''}
    for entity, char in html_entities.items(): text = text.replace(entity, char)
    return text

def normalize_title_for_comparison(title):
    if not isinstance(title, str): return ""
    title = convert_html_entities(title)
    return re.sub(r'\W+', ' ', title).lower().strip()

def clean_title_for_output(title):
    if not isinstance(title, str): return ""
    title = convert_html_entities(title)
    title = re.sub(r'\s*\|\s*[\w\s]+$', '', title).strip()
    return title

def corregir_texto(text):
    if not isinstance(text, str): return text
    text = convert_html_entities(text)
    text = re.sub(r'(<br>|\[\.\.\.\]|\s+)', ' ', text).strip()
    match = re.search(r'[A-Z]', text)
    if match: text = text[match.start():]
    if text and not text.endswith('...'): text = text.rstrip('.') + '...'
    return text

def preprocess_text_for_topic(text: str) -> str:
    if not isinstance(text, str): return ""
    from nltk.corpus import stopwords
    stop_words_list = set(stopwords.words('spanish'))
    token_pattern_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    tokens = token_pattern_re.findall(text.lower())
    return " ".join(tok for tok in tokens if tok not in stop_words_list)

def to_excel_from_df(df, original_headers):
    output = io.BytesIO()
    # Reordenar columnas según el orden original y añadir las nuevas
    final_order = original_headers + [col for col in ['Duplicada', 'Posible Duplicada', 'Mantener'] if col in df.columns]
    # Asegurarse de que todas las columnas existen en el df antes de reordenar
    final_columns_in_df = [col for col in final_order if col in df.columns]
    df_to_excel = df[final_columns_in_df]
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_excel.to_excel(writer, index=False, sheet_name='Resultado')
    return output.getvalue()

@st.cache_resource
def load_ml_models():
    sentiment_model = joblib.load('modelo_sentimiento.pkl')
    sentiment_vectorizer = joblib.load('vectorizador_sentimiento.pkl')
    topic_model = joblib.load('modelo_tema.pkl')
    topic_vectorizer = joblib.load('vectorizador_tema.pkl')
    return sentiment_model, sentiment_vectorizer, topic_model, topic_vectorizer

# ==============================================================================
# LÓGICA DE PROCESAMIENTO PRINCIPAL
# ==============================================================================
def run_full_process(dossier_file, config_file):
    
    # --- PASO 1: CARGAR DATOS Y MODELOS ---
    st.markdown("---")
    progress_text = st.empty()
    
    progress_text.info("Paso 1/8: Cargando modelos de IA y archivos de configuración...")
    try:
        sentiment_model, sentiment_vectorizer, topic_model, topic_vectorizer = load_ml_models()
        
        # --- SOLUCIÓN AL ValueError: Usar .read() para pasar los bytes a Pandas ---
        config_sheets = pd.read_excel(config_file.read(), sheet_name=None)
        
        region_map = pd.Series(config_sheets['Regiones'].iloc[:, 1].values, index=config_sheets['Regiones'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        internet_map = pd.Series(config_sheets['Internet'].iloc[:, 1].values, index=config_sheets['Internet'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        mention_map = pd.Series(config_sheets['Menciones'].iloc[:, 1].values, index=config_sheets['Menciones'].iloc[:, 0].astype(str).str.strip()).to_dict()
        final_topic_map = pd.Series(config_sheets['Mapa_Temas'].iloc[:, 1].values, index=config_sheets['Mapa_Temas'].iloc[:, 0].astype(str).str.strip()).to_dict()
        
    except FileNotFoundError as e:
        st.error(f"Error fatal: No se pudo cargar un archivo de modelo. Asegúrate que los archivos .pkl estén en la misma carpeta. Archivo: {e.filename}")
        st.stop()
    except KeyError as e:
        st.error(f"Error en el archivo de Configuración: Falta la hoja '{e}'. Por favor, revisa el archivo `Configuracion.xlsx`.")
        st.stop()

    # --- PASO 2: LIMPIEZA INICIAL Y EXPANSIÓN DE MENCIONES ---
    progress_text.info("Paso 2/8: Realizando limpieza inicial y expansión de filas...")
    wb = load_workbook(dossier_file)
    sheet = wb.active
    original_headers = [cell.value for cell in sheet[1] if cell.value]
    
    rows_to_expand = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if all(c is None for c in row): continue
        row_data = dict(zip(original_headers, row))
        menciones = [m.strip() for m in str(row_data.get('Menciones - Empresa') or '').split(';') if m.strip()]
        if not menciones:
            rows_to_expand.append(row_data)
        else:
            for mencion in menciones:
                new_row = row_data.copy()
                new_row['Menciones - Empresa'] = mencion
                rows_to_expand.append(new_row)
    
    df = pd.DataFrame(rows_to_expand)
    df['Mantener'] = 'Conservar'

    # --- PASO 3: LIMPIEZA Y MAPEADO CON PANDAS ---
    progress_text.info("Paso 3/8: Aplicando mapeos y normalizaciones...")
    df['Título'] = df['Título'].astype(str).apply(clean_title_for_output)
    df['Resumen - Aclaracion'] = df['Resumen - Aclaracion'].astype(str).apply(corregir_texto)
    
    df['Región'] = df['Medio'].astype(str).str.lower().str.strip().map(region_map)
    df['Menciones - Empresa'] = df['Menciones - Empresa'].astype(str).str.strip().map(mention_map).fillna(df['Menciones - Empresa'])
    is_internet = df['Tipo de Medio'].astype(str).str.lower().str.strip() == 'internet'
    df.loc[is_internet, 'Medio'] = df.loc[is_internet, 'Medio'].astype(str).str.lower().str.strip().map(internet_map).fillna(df.loc[is_internet, 'Medio'])

    # --- PASO 4: DETECCIÓN DE DUPLICADOS ---
    progress_text.info("Paso 4/8: Detectando y marcando duplicados...")
    df['titulo_norm'] = df['Título'].apply(normalize_title_for_comparison)
    dup_cols = ['titulo_norm', 'Medio', 'Fecha', 'Menciones - Empresa']
    df_dups = df[df.duplicated(subset=dup_cols, keep='first')]
    df.loc[df_dups.index, 'Mantener'] = 'Eliminar'
    df.loc[df_dups.index, ['Tono', 'Tema', 'Temas Generales - Tema']] = 'Duplicada'
    
    # --- PASO 5: INFERENCIA DE IA ---
    progress_text.info("Paso 5/8: Aplicando modelos de IA para Tono y Tema...")
    df_valid = df[df['Mantener'] == 'Conservar'].copy()
    if not df_valid.empty:
        df_valid['texto_para_ia'] = df_valid['Título'].fillna('') + ' ' + df_valid['Resumen - Aclaracion'].fillna('')
        
        X_sent = sentiment_vectorizer.transform(df_valid['texto_para_ia'])
        preds_sent = sentiment_model.predict(X_sent)
        label_map_inv = {1: 'Positivo', 0: 'Neutro', -1: 'Negativo'}
        df_valid['Tono'] = [label_map_inv.get(p, 'Indefinido') for p in preds_sent]
        
        df_valid["resumen_procesado"] = df_valid["texto_para_ia"].apply(preprocess_text_for_topic)
        X_tema = topic_vectorizer.transform(df_valid["resumen_procesado"])
        df_valid["Temas Generales - Tema"] = topic_model.predict(X_tema)
        
        df.update(df_valid[['Tono', 'Temas Generales - Tema']])

    # --- PASO 6: HOMOGENEIZACIÓN DE TEMAS POR TÍTULO ---
    progress_text.info("Paso 6/8: Homogeneizando temas para mayor consistencia...")
    if 'Temas Generales - Tema' in df.columns:
        df['Temas Generales - Tema'] = df.groupby('titulo_norm')['Temas Generales - Tema'].transform(lambda x: x.mode()[0] if not x.mode().empty else x)

    # --- PASO 7: MAPEO FINAL DE TEMA ---
    progress_text.info("Paso 7/8: Realizando el mapeo final de Tema...")
    if 'Temas Generales - Tema' in df.columns:
        df['Tema'] = df['Temas Generales - Tema'].astype(str).str.strip().map(final_topic_map).fillna('Indefinido')

    # --- PASO 8: PREPARACIÓN FINAL Y ESTADÍSTICAS ---
    progress_text.info("Paso 8/8: Generando resultados y estadísticas...")
    df.drop(columns=['titulo_norm'], inplace=True, errors='ignore')
    
    st.balloons()
    progress_text.success("¡Proceso completado con éxito!")

    st.subheader("📊 Resumen del Proceso")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas Totales Procesadas", len(df))
    dups_count = (df['Mantener'] == 'Eliminar').sum()
    col2.metric("Duplicados Eliminados", dups_count)
    col3.metric("Filas Finales", len(df) - dups_count)

    with st.expander("Ver alertas de calidad de datos"):
        medios_sin_region = df[df['Región'].isnull()]['Medio'].unique()
        if len(medios_sin_region) > 0:
            st.warning(f"**{len(medios_sin_region)} medios no encontrados en el mapeo de Regiones:**")
            st.code('\n'.join(medios_sin_region[:10]))
        else:
            st.success("Todos los medios fueron mapeados a una región.")

    st.subheader("✍️ Previsualización y Edición de Resultados")
    st.info("Puedes editar los datos directamente en la tabla. Los cambios se guardarán en el archivo descargado.")
    
    edited_df = st.data_editor(df, num_rows="dynamic", key="final_editor", use_container_width=True)
    
    excel_data = to_excel_from_df(edited_df, original_headers)
    st.download_button(
        label="📥 Descargar Archivo Final Procesado",
        data=excel_data,
        file_name=f"Dossier_Procesado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("🚀 Procesador Inteligente de Dossiers v2.1")
st.markdown("Una herramienta para limpiar, enriquecer y analizar dossieres de noticias de forma automática.")

st.info(
    "**Instrucciones:**\n\n"
    "1. Prepara tu archivo **Dossier** principal en formato `.xlsx`.\n"
    "2. Prepara tu archivo de mapeos llamado **`Configuracion.xlsx`**.\n"
    "3. Sube ambos archivos juntos en el área de abajo.\n"
    "4. Haz clic en 'Iniciar Proceso'."
)

with st.expander("Ver estructura requerida para `Configuracion.xlsx`"):
    st.markdown("""
    Tu archivo `Configuracion.xlsx` debe contener exactamente estas 4 hojas:
    - **`Regiones`**: Columna A (Medio), Columna B (Región).
    - **`Internet`**: Columna A (Medio Original), Columna B (Medio Mapeado).
    - **`Menciones`**: Columna A (Mención Original), Columna B (Mención Mapeada).
    - **`Mapa_Temas`**: Columna A (Temas Generales - Tema), Columna B (Tema).
    """)

uploaded_files = st.file_uploader(
    "Arrastra y suelta tus archivos aquí (Dossier y Configuracion)",
    type=["xlsx"],
    accept_multiple_files=True
)

dossier_file = None
config_file = None

if uploaded_files:
    for file in uploaded_files:
        if 'config' in file.name.lower():
            config_file = file
        else:
            dossier_file = file

    if dossier_file:
        st.success(f"Archivo Dossier cargado: **{dossier_file.name}**")
    else:
        st.warning("No se ha subido un archivo que parezca ser el Dossier.")
        
    if config_file:
        st.success(f"Archivo de Configuración cargado: **{config_file.name}**")
    else:
        st.warning("No se ha subido el archivo `Configuracion.xlsx`.")

if st.button("▶️ Iniciar Proceso Completo", disabled=not (dossier_file and config_file), type="primary"):
    run_full_process(dossier_file, config_file)
