import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import datetime
import joblib
import numpy as np
import nltk

# Importar todas nuestras funciones de ayuda desde el nuevo archivo
import dossier_utils as utils

# --- Configuración de la página ---
st.set_page_config(page_title="Procesador de Dossiers Nissan v3.9", layout="wide")

# --- Descarga NLTK stopwords si es necesario ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Descargando recursos de lenguaje por primera vez...")
    nltk.download('stopwords')
    st.success("Recursos listos.")

# ==============================================================================
# FUNCIONES DE CARGA Y PROCESO PRINCIPAL
# ==============================================================================

@st.cache_resource
def load_ml_models():
    """Carga los pipelines completos de sentimiento y tema."""
    try:
        sentiment_pipeline = joblib.load('pipeline_sentimiento.pkl')
        topic_pipeline = joblib.load('pipeline_tema.pkl')
        return sentiment_pipeline, topic_pipeline
    except FileNotFoundError as e:
        st.error(f"Error Crítico: No se encontró el archivo de modelo: {e.filename}. Asegúrate de que 'pipeline_sentimiento.pkl' y 'pipeline_tema.pkl' estén en la misma carpeta que la aplicación.")
        st.stop()

def read_and_expand_dossier(dossier_file):
    """Lee el archivo Excel, extrae hyperlinks y expande las filas por mención."""
    wb = load_workbook(dossier_file)
    sheet = wb.active
    original_headers = [cell.value for cell in sheet[1] if cell.value]
    
    rows_data = []
    # Itera sobre las filas para extraer datos y links correctamente
    for row in sheet.iter_rows(min_row=2, values_only=False):
        if all(c.value is None for c in row): continue
        
        row_values = {}
        for i, header in enumerate(original_headers):
            # Lógica especial para extraer el link o el valor de la celda
            if header in ['Link Nota', 'Link (Streaming - Imagen)']:
                row_values[header] = utils.extract_link_from_cell(row[i])
            else:
                row_values[header] = row[i].value
        
        # Expande las filas por cada mención separada por ';'
        menciones_str = str(row_values.get('Menciones - Empresa') or '')
        menciones = [m.strip() for m in menciones_str.split(';') if m.strip()]
        
        if not menciones:
            rows_data.append(row_values)
        else:
            for mencion in menciones:
                new_row = row_values.copy()
                new_row['Menciones - Empresa'] = mencion
                rows_data.append(new_row)
    
    return pd.DataFrame(rows_data), original_headers

def run_full_process(dossier_file, config_file):
    
    st.markdown("---")
    progress_bar = st.progress(0, text="Iniciando proceso...")

    # --- 1. Carga de modelos y configuración ---
    progress_bar.progress(5, text="Paso 1/8: Cargando modelos y configuración...")
    sentiment_pipeline, topic_pipeline = load_ml_models()
    try:
        config_sheets = pd.read_excel(config_file, sheet_name=None)
        region_map = pd.Series(config_sheets['Regiones'].iloc[:, 1].values, index=config_sheets['Regiones'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        internet_map = pd.Series(config_sheets['Internet'].iloc[:, 1].values, index=config_sheets['Internet'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        mention_map = pd.Series(config_sheets['Menciones'].iloc[:, 1].values, index=config_sheets['Menciones'].iloc[:, 0].astype(str).str.strip()).to_dict()
        final_topic_map = pd.Series(config_sheets['Mapa_Temas'].iloc[:, 1].values, index=config_sheets['Mapa_Temas'].iloc[:, 0].astype(str).str.strip()).to_dict()
    except Exception as e:
        st.error(f"Error al cargar `Configuracion.xlsx`: {e}. Revisa que el archivo y sus hojas sean correctos.")
        st.stop()

    # --- 2. Lectura y Expansión del Dossier ---
    progress_bar.progress(15, text="Paso 2/8: Leyendo Dossier y expandiendo filas...")
    df, original_headers = read_and_expand_dossier(dossier_file)

    # --- 3. Limpieza y Normalización de Datos ---
    progress_bar.progress(25, text="Paso 3/8: Aplicando mapeos y normalizaciones...")
    
    # *** LA SOLUCIÓN CLAVE PARA LAS FECHAS ***
    if 'Fecha' in df.columns:
        original_dates = df['Fecha'].copy()
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True)
        failed_dates = df['Fecha'].isna().sum()
        if failed_dates > 0:
            st.warning(f"⚠️ **Atención:** No se pudieron convertir {failed_dates} fechas. Se mantendrán como vacías. Revisa el archivo original si esto es inesperado.")

    # Limpieza de texto usando funciones de utils
    df['Título Limpio'] = df['Título'].apply(utils.clean_title_for_output)
    df['Resumen - Aclaracion'] = df['Resumen - Aclaracion'].apply(utils.corregir_resumen)
    
    # Mapeos
    tipo_medio_map = {'online': 'Internet', 'diario': 'Prensa', 'am': 'Radio', 'fm': 'Radio', 'aire': 'Televisión', 'cable': 'Televisión', 'revista': 'Revista'}
    df['Tipo de Medio'] = df['Tipo de Medio'].str.lower().str.strip().map(tipo_medio_map).fillna(df['Tipo de Medio'])
    df['Región'] = df['Medio'].astype(str).str.lower().str.strip().map(region_map)
    df['Menciones - Empresa'] = df['Menciones - Empresa'].astype(str).str.strip().map(mention_map).fillna(df['Menciones - Empresa'])
    
    is_internet = df['Tipo de Medio'] == 'Internet'
    df.loc[is_internet, 'Medio'] = df.loc[is_internet, 'Medio'].astype(str).str.lower().str.strip().map(internet_map).fillna(df.loc[is_internet, 'Medio'])

    # --- 4. Reorganización de Columnas Específicas ---
    progress_bar.progress(40, text="Paso 4/8: Reorganizando columnas de links y dimensiones...")
    is_print = df['Tipo de Medio'].isin(['Prensa', 'Revista'])
    is_broadcast = df['Tipo de Medio'].isin(['Radio', 'Televisión'])
    
    # Intercambiar links para Internet si es necesario
    df.loc[is_internet, ['Link Nota', 'Link (Streaming - Imagen)']] = df.loc[is_internet, ['Link (Streaming - Imagen)', 'Link Nota']].values
    # Copiar link para Prensa si 'Link Nota' está vacío
    cond_copy = is_print & df['Link Nota'].isnull() & df['Link (Streaming - Imagen)'].notnull()
    df.loc[cond_copy, 'Link Nota'] = df.loc[cond_copy, 'Link (Streaming - Imagen)']
    df.loc[is_print | is_broadcast, 'Link (Streaming - Imagen)'] = None # Limpiar link de streaming para no-internet
    
    # Mover 'Duración' a 'Dimensión' para Radio/TV
    if 'Duración - Nro. Caracteres' in df.columns and 'Dimensión' in df.columns:
        df.loc[is_broadcast, 'Dimensión'] = df.loc[is_broadcast, 'Duración - Nro. Caracteres']
        df.loc[is_broadcast, 'Duración - Nro. Caracteres'] = np.nan

    # --- 5. Detección de Duplicados ---
    progress_bar.progress(50, text="Paso 5/8: Detectando duplicados con criterio de calidad...")
    df_reset = df.reset_index().rename(columns={'index': 'original_index'})
    df_reset['title_quality'] = df_reset['Título'].apply(utils.calculate_title_quality_score)
    df_reset.sort_values(by=['title_quality', 'Fecha', 'original_index'], ascending=[False, True, True], inplace=True)
    
    rows_list = df_reset.to_dict('records')
    is_duplicate_map = {}
    for i in range(len(rows_list)):
        if rows_list[i]['original_index'] in is_duplicate_map: continue
        for j in range(i + 1, len(rows_list)):
            if rows_list[j]['original_index'] in is_duplicate_map: continue
            
            if utils.are_duplicates(pd.Series(rows_list[i]), pd.Series(rows_list[j])):
                is_duplicate_map[rows_list[j]['original_index']] = True
    
    df_reset['is_duplicate'] = df_reset['original_index'].map(is_duplicate_map).fillna(False)
    df = df_reset.sort_values('original_index').set_index('original_index')

    # --- 6. Aplicación de Modelos de IA ---
    progress_bar.progress(70, text="Paso 6/8: Aplicando modelos de IA a noticias únicas...")
    df_valid = df[~df['is_duplicate']].copy()
    if not df_valid.empty:
        df_valid['texto_para_ia'] = df_valid['Título Limpio'].fillna('') + ' ' + df_valid['Resumen - Aclaracion'].fillna('')
        
        preds_sent = sentiment_pipeline.predict(df_valid['texto_para_ia'])
        label_map_inv = {1: 'Positivo', 0: 'Neutro', -1: 'Negativo'}
        df_valid['Tono'] = [label_map_inv.get(p, 'Indefinido') for p in preds_sent]
        
        df_valid["resumen_procesado"] = df_valid["texto_para_ia"].apply(utils.preprocess_text_for_topic)
        df_valid["Temas Generales - Tema"] = topic_pipeline.predict(df_valid["resumen_procesado"])
        
        df.update(df_valid[['Tono', 'Temas Generales - Tema']])

    # --- 7. Homogeneización de Temas y Mapeo Final ---
    progress_bar.progress(85, text="Paso 7/8: Homogeneizando y mapeando temas...")
    df_valid_homog = df[~df['is_duplicate']].copy()
    if not df_valid_homog.empty and 'Temas Generales - Tema' in df_valid_homog.columns:
        df_valid_homog['titulo_norm_homog'] = df_valid_homog['Título'].apply(utils.normalize_title_for_comparison)
        homogenized_temas = df_valid_homog.groupby('titulo_norm_homog')['Temas Generales - Tema'].transform(lambda x: x.mode()[0] if not x.mode().empty else x)
        df_valid_homog['Temas Generales - Tema'] = homogenized_temas
        df.update(df_valid_homog[['Temas Generales - Tema']])

    if 'Temas Generales - Tema' in df.columns:
        df['Tema'] = df['Temas Generales - Tema'].astype(str).str.strip().map(final_topic_map).fillna('Indefinido')
    
    df.loc[df['is_duplicate'], ['Tono', 'Tema', 'Temas Generales - Tema']] = 'Duplicada'
    df['Título'] = df['Título Limpio']

    # --- 8. Generación de Resultados Finales ---
    progress_bar.progress(100, text="Paso 8/8: ¡Proceso completado!")
    st.balloons()
    
    final_order = ["ID Noticia", "Fecha", "Hora", "Medio", "Tipo de Medio", "Sección - Programa", "Región", "Título", "Autor - Conductor", "Nro. Pagina", "Dimensión", "Duración - Nro. Caracteres", "CPE", "Tier", "Audiencia", "Tono", "Tema", "Temas Generales - Tema", "Resumen - Aclaracion", "Link Nota", "Link (Streaming - Imagen)", "Menciones - Empresa"]
    
    st.subheader("📊 Resumen del Proceso")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas Totales Procesadas", len(df))
    dups_count = df['is_duplicate'].sum()
    col2.metric("Filas Marcadas como Duplicadas", dups_count)
    col3.metric("Filas Únicas Analizadas", len(df) - dups_count)
    
    excel_data = utils.to_excel_from_df(df, final_order)
    st.download_button(label="📥 Descargar Archivo Final Procesado", data=excel_data, file_name=f"Dossier_Procesado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.sheet")

    st.subheader("✍️ Previsualización de Resultados")
    final_cols_in_df = [col for col in final_order if col in df.columns]
    df_display = df[final_cols_in_df].copy()
    
    # Formateo final para la visualización en Streamlit
    if 'Fecha' in df_display.columns:
        df_display['Fecha'] = pd.to_datetime(df_display['Fecha']).dt.strftime('%d/%m/%Y').replace('NaT', 'FECHA INVÁLIDA')
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("🚀 Procesador Inteligente de Dossiers v3.9")
st.markdown("Herramienta para limpiar, enriquecer y analizar dossieres de noticias. **Versión con manejo de fechas mejorado y código refactorizado.**")
st.info("**Instrucciones:**\n\n1. Prepara tu archivo **Dossier** principal y tu archivo **`Configuracion.xlsx`**.\n2. Sube ambos archivos en el área de abajo.\n3. Haz clic en '▶️ Iniciar Proceso Completo'.")

with st.expander("Ver estructura requerida para `Configuracion.xlsx`"):
    st.markdown("- **`Regiones`**: Columna A (Medio), Columna B (Región).\n- **`Internet`**: Columna A (Medio Original), Columna B (Medio Mapeado).\n- **`Menciones`**: Columna A (Mención Original), Columna B (Mención Mapeada).\n- **`Mapa_Temas`**: Columna A (Temas Generales - Tema), Columna B (Tema).")

uploaded_files = st.file_uploader("Arrastra y suelta tus archivos aquí (Dossier y Configuracion.xlsx)", type=["xlsx"], accept_multiple_files=True)
dossier_file, config_file = None, None
if uploaded_files:
    for file in uploaded_files:
        if 'config' in file.name.lower():
            config_file = file
        else:
            dossier_file = file
    
    if dossier_file: st.success(f"✅ Archivo Dossier cargado: **{dossier_file.name}**")
    else: st.warning("⚠️ No se ha subido un archivo que parezca ser el Dossier.")
    
    if config_file: st.success(f"✅ Archivo de Configuración cargado: **{config_file.name}**")
    else: st.warning("⚠️ No se ha subido el archivo `Configuracion.xlsx`.")

if st.button("▶️ Iniciar Proceso Completo", disabled=not (dossier_file and config_file), type="primary"):
    run_full_process(dossier_file, config_file)
