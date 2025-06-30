import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import datetime
import io
import joblib
import re
import nltk

# --- Configuración de la página ---
st.set_page_config(page_title="Procesador de Dossiers Nissan v2.5", layout="wide")

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
def extract_link_from_cell(cell):
    """Extrae el hipervínculo de una celda de openpyxl."""
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target
    return None

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

def to_excel_from_df(df, final_order):
    output = io.BytesIO()
    final_columns_in_df = [col for col in final_order if col in df.columns]
    df_to_excel = df[final_columns_in_df]

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_excel.to_excel(writer, index=False, sheet_name='Resultado')
        workbook = writer.book
        worksheet = writer.sheets['Resultado']
        
        link_format = workbook.add_format({'color': 'blue', 'underline': 1})
        
        for col_name in ['Link Nota', 'Link (Streaming - Imagen)']:
            if col_name in df_to_excel.columns:
                col_idx = df_to_excel.columns.get_loc(col_name)
                for row_idx, url in enumerate(df_to_excel[col_name]):
                    if pd.notna(url) and isinstance(url, str) and url.startswith('http'):
                        worksheet.write_url(row_idx + 1, col_idx, url, link_format, 'Link')
                        
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
    
    st.markdown("---")
    progress_text = st.empty()
    
    progress_text.info("Paso 1/8: Cargando modelos y configuración...")
    try:
        sentiment_model, sentiment_vectorizer, topic_model, topic_vectorizer = load_ml_models()
        config_sheets = pd.read_excel(config_file.read(), sheet_name=None)
        
        region_map = pd.Series(config_sheets['Regiones'].iloc[:, 1].values, index=config_sheets['Regiones'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        internet_map = pd.Series(config_sheets['Internet'].iloc[:, 1].values, index=config_sheets['Internet'].iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        mention_map = pd.Series(config_sheets['Menciones'].iloc[:, 1].values, index=config_sheets['Menciones'].iloc[:, 0].astype(str).str.strip()).to_dict()
        final_topic_map = pd.Series(config_sheets['Mapa_Temas'].iloc[:, 1].values, index=config_sheets['Mapa_Temas'].iloc[:, 0].astype(str).str.strip()).to_dict()
        
    except FileNotFoundError as e:
        st.error(f"Error: Archivo de modelo no encontrado: {e.filename}. Asegúrate que los archivos .pkl estén en la misma carpeta.")
        st.stop()
    except KeyError as e:
        st.error(f"Error: Falta la hoja '{e}' en `Configuracion.xlsx`.")
        st.stop()

    progress_text.info("Paso 2/8: Leyendo Dossier y expandiendo filas...")
    wb = load_workbook(dossier_file)
    sheet = wb.active
    original_headers = [cell.value for cell in sheet[1] if cell.value]
    link_nota_idx = original_headers.index('Link Nota') if 'Link Nota' in original_headers else -1
    link_stream_idx = original_headers.index('Link (Streaming - Imagen)') if 'Link (Streaming - Imagen)' in original_headers else -1

    rows_to_expand = []
    for row_idx, row in enumerate(sheet.iter_rows(min_row=2)):
        if all(c.value is None for c in row): continue
        
        row_values = [cell.value for cell in row]
        row_data = dict(zip(original_headers, row_values))

        if link_nota_idx != -1: row_data['Link Nota'] = extract_link_from_cell(row[link_nota_idx])
        if link_stream_idx != -1: row_data['Link (Streaming - Imagen)'] = extract_link_from_cell(row[link_stream_idx])

        menciones = [m.strip() for m in str(row_data.get('Menciones - Empresa') or '').split(';') if m.strip()]
        if not menciones: rows_to_expand.append(row_data)
        else:
            for mencion in menciones:
                new_row = row_data.copy()
                new_row['Menciones - Empresa'] = mencion
                rows_to_expand.append(new_row)
    
    df = pd.DataFrame(rows_to_expand)
    df['Mantener'] = 'Conservar'

    progress_text.info("Paso 3/8: Aplicando mapeos y normalizaciones...")
    for col in original_headers:
        if col not in df.columns: df[col] = None
            
    df['Título'] = df['Título'].astype(str).apply(clean_title_for_output)
    df['Resumen - Aclaracion'] = df['Resumen - Aclaracion'].astype(str).apply(corregir_texto)

    # --- INICIO DE LA FUNCIONALIDAD RESTAURADA ---
    tipo_medio_map = {'online': 'Internet', 'diario': 'Prensa', 'am': 'Radio', 'fm': 'Radio', 'aire': 'Televisión', 'cable': 'Televisión', 'revista': 'Revista'}
    df['Tipo de Medio'] = df['Tipo de Medio'].str.lower().str.strip().map(tipo_medio_map).fillna(df['Tipo de Medio'])
    # --- FIN DE LA FUNCIONALIDAD RESTAURADA ---
    
    df['Región'] = df['Medio'].astype(str).str.lower().str.strip().map(region_map)
    df['Menciones - Empresa'] = df['Menciones - Empresa'].astype(str).str.strip().map(mention_map).fillna(df['Menciones - Empresa'])
    is_internet = df['Tipo de Medio'].astype(str).str.lower().str.strip() == 'internet'
    df.loc[is_internet, 'Medio'] = df.loc[is_internet, 'Medio'].astype(str).str.lower().str.strip().map(internet_map).fillna(df.loc[is_internet, 'Medio'])

    progress_text.info("Paso 4/8: Detectando y marcando duplicados...")
    df['titulo_norm'] = df['Título'].apply(normalize_title_for_comparison)
    dup_cols = ['titulo_norm', 'Medio', 'Fecha', 'Menciones - Empresa']
    duplicated_mask = df.duplicated(subset=dup_cols, keep='first')
    df.loc[duplicated_mask, 'Mantener'] = 'Eliminar'
    
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

    progress_text.info("Paso 6/8: Homogeneizando temas para mayor consistencia...")
    df_valid_homog = df[df['Mantener'] == 'Conservar'].copy()
    if not df_valid_homog.empty and 'Temas Generales - Tema' in df_valid_homog.columns:
        homogenized_temas = df_valid_homog.groupby('titulo_norm')['Temas Generales - Tema'].transform(lambda x: x.mode()[0] if not x.mode().empty else x)
        df_valid_homog['Temas Generales - Tema'] = homogenized_temas
        df.update(df_valid_homog[['Temas Generales - Tema']])

    progress_text.info("Paso 7/8: Realizando el mapeo final de Tema...")
    if 'Temas Generales - Tema' in df.columns:
        df['Tema'] = df['Temas Generales - Tema'].astype(str).str.strip().map(final_topic_map).fillna('Indefinido')

    df.loc[df['Mantener'] == 'Eliminar', ['Tono', 'Tema', 'Temas Generales - Tema']] = 'Duplicada'
    
    progress_text.info("Paso 8/8: Generando resultados finales...")
    st.balloons()
    progress_text.success("¡Proceso completado con éxito!")

    final_order = [
        "ID Noticia", "Fecha", "Hora", "Medio", "Tipo de Medio", "Sección - Programa", 
        "Región", "Título", "Autor - Conductor", "Nro. Pagina", "Dimensión", 
        "Duración - Nro. Caracteres", "CPE", "Tier", "Audiencia", "Tono", "Tema", 
        "Temas Generales - Tema", "Resumen - Aclaracion", "Link Nota", 
        "Link (Streaming - Imagen)", "Menciones - Empresa"
    ]
    
    df_final = df.copy()
    df_final = df_final.reset_index(drop=True)

    st.subheader("📊 Resumen del Proceso")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas Totales", len(df_final))
    dups_count = (df_final['Mantener'] == 'Eliminar').sum()
    col2.metric("Filas Marcadas como Duplicadas", dups_count)
    col3.metric("Filas Únicas", len(df_final) - dups_count)

    st.subheader("✍️ Previsualización y Edición de Resultados")
    st.info("Puedes editar los datos directamente en la tabla. Los cambios se guardarán en el archivo descargado.")
    
    final_columns_in_df = [col for col in final_order if col in df_final.columns]
    
    df_for_editor = df_final[final_columns_in_df].copy()
    for col_name in ['Link Nota', 'Link (Streaming - Imagen)']:
        if col_name in df_for_editor.columns:
            df_for_editor[col_name] = df_for_editor[col_name].apply(lambda x: 'Link' if pd.notna(x) else '')
            
    edited_df_display = st.data_editor(df_for_editor, num_rows="dynamic", key="final_editor", use_container_width=True)
    
    df_to_download = edited_df_display.copy()
    for col_name in ['Link Nota', 'Link (Streaming - Imagen)']:
        if col_name in df_to_download.columns:
            df_to_download[col_name] = df_final[col_name]

    excel_data = to_excel_from_df(df_to_download, final_order)
    st.download_button(
        label="📥 Descargar Archivo Final Procesado",
        data=excel_data,
        file_name=f"Dossier_Procesado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("🚀 Procesador Inteligente de Dossiers v2.5")
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
