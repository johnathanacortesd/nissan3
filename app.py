import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import datetime
import io
import joblib
import re
import nltk
import html
import numpy as np
from difflib import SequenceMatcher

# --- Configuración de la página ---
st.set_page_config(page_title="Procesador de Dossiers Nissan v3.6", layout="wide")

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
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target
    return None

def convert_html_entities(text):
    if not isinstance(text, str): return text
    text = html.unescape(text)
    custom_replacements = { '“': '\"', '”': '\"', '‘': "'", '’': "'", 'Â': '', 'â': '', '€': '', '™': '' }
    for entity, char in custom_replacements.items():
        text = text.replace(entity, char)
    return text

def normalize_title_for_comparison(title):
    if not isinstance(title, str):
        return ""
    cleaned_title = convert_html_entities(title)
    cleaned_title = re.sub(r'\s*[|-].*$', '', cleaned_title).strip()
    normalized_title = re.sub(r'\W+', ' ', cleaned_title).lower().strip()
    return normalized_title

def clean_title_for_output(title):
    if not isinstance(title, str): return ""
    title = convert_html_entities(title)
    title = re.sub(r'\s*[|-].*$', '', title).strip()
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
    with pd.ExcelWriter(
        output,
        engine='xlsxwriter',
        datetime_format='dd/mm/yyyy',
        date_format='dd/mm/yyyy'
    ) as writer:
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

def are_duplicates(row1, row2, title_similarity_threshold=0.85, date_proximity_days=1):
    if row1['Menciones - Empresa'] != row2['Menciones - Empresa']:
        return False
    if row1['Medio'] != row2['Medio']:
        return False

    titulo1 = normalize_title_for_comparison(row1['Título'])
    titulo2 = normalize_title_for_comparison(row2['Título'])

    try:
        fecha1 = row1['Fecha'].date() if pd.notna(row1['Fecha']) else None
        fecha2 = row2['Fecha'].date() if pd.notna(row2['Fecha']) else None
        if fecha1 is None or fecha2 is None: return False
    except AttributeError:
        fecha1 = row1['Fecha']
        fecha2 = row2['Fecha']
    except Exception:
        return False

    if row1['Tipo de Medio'] == 'Internet':
        if row1['Hora'] == row2['Hora']: return False
        if abs((fecha1 - fecha2).days) > date_proximity_days: return False
        
        if titulo1 == titulo2 and titulo1 != "": return True
        similarity = SequenceMatcher(None, titulo1, titulo2).ratio()
        if similarity >= title_similarity_threshold: return True
    else:
        if row1['Tipo de Medio'] in ['Radio', 'Televisión']:
            if row1['Hora'] != row2['Hora']: return False
        if fecha1 != fecha2: return False
        if titulo1 == titulo2 and titulo1 != "": return True
            
    return False

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
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}. Revisa que los archivos de modelos (.pkl) y `Configuracion.xlsx` sean correctos.")
        st.stop()

    progress_text.info("Paso 2/8: Leyendo Dossier y expandiendo filas...")
    wb = load_workbook(dossier_file)
    sheet = wb.active
    
    # --- LÍNEA CORREGIDA ---
    original_headers = [cell.value for cell in sheet[1] if cell.value]
    
    rows_to_expand = []
    for row in sheet.iter_rows(min_row=2):
        if all(c.value is None for c in row): continue
        row_values = [c.value for c in row]
        row_data = dict(zip(original_headers, row_values))
        if 'Link Nota' in original_headers:
            link_nota_index = original_headers.index('Link Nota')
            row_data['Link Nota'] = extract_link_from_cell(row[link_nota_index])
        if 'Link (Streaming - Imagen)' in original_headers:
            link_streaming_index = original_headers.index('Link (Streaming - Imagen)')
            row_data['Link (Streaming - Imagen)'] = extract_link_from_cell(row[link_streaming_index])
        
        menciones = [m.strip() for m in str(row_data.get('Menciones - Empresa') or '').split(';') if m.strip()]
        if not menciones:
            rows_to_expand.append(row_data)
        else:
            for mencion in menciones:
                new_row = row_data.copy()
                new_row['Menciones - Empresa'] = mencion
                rows_to_expand.append(new_row)
    df = pd.DataFrame(rows_to_expand)
    
    progress_text.info("Paso 3/8: Aplicando mapeos y normalizaciones...")
    for col in original_headers:
        if col not in df.columns: df[col] = None
    
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Título'] = df['Título'].astype(str).apply(clean_title_for_output)
    df['Resumen - Aclaracion'] = df['Resumen - Aclaracion'].astype(str).apply(corregir_texto)
    tipo_medio_map = {'online': 'Internet', 'diario': 'Prensa', 'am': 'Radio', 'fm': 'Radio', 'aire': 'Televisión', 'cable': 'Televisión', 'revista': 'Revista'}
    df['Tipo de Medio'] = df['Tipo de Medio'].str.lower().str.strip().map(tipo_medio_map).fillna(df['Tipo de Medio'])
    
    is_internet = df['Tipo de Medio'] == 'Internet'
    is_print = df['Tipo de Medio'].isin(['Prensa', 'Revista'])
    is_broadcast = df['Tipo de Medio'].isin(['Radio', 'Televisión'])
    
    df.loc[is_internet, ['Link Nota', 'Link (Streaming - Imagen)']] = df.loc[is_internet, ['Link (Streaming - Imagen)', 'Link Nota']].values
    cond_copy = is_print & df['Link Nota'].isnull() & df['Link (Streaming - Imagen)'].notnull()
    df.loc[cond_copy, 'Link Nota'] = df.loc[cond_copy, 'Link (Streaming - Imagen)']
    df.loc[is_print | is_broadcast, 'Link (Streaming - Imagen)'] = None
    
    if 'Duración - Nro. Caracteres' in df.columns and 'Dimensión' in df.columns:
        df.loc[is_broadcast, 'Dimensión'] = df.loc[is_broadcast, 'Duración - Nro. Caracteres']
        df.loc[is_broadcast, 'Duración - Nro. Caracteres'] = np.nan

    df['Región'] = df['Medio'].astype(str).str.lower().str.strip().map(region_map)
    df['Menciones - Empresa'] = df['Menciones - Empresa'].astype(str).str.strip().map(mention_map).fillna(df['Menciones - Empresa'])
    df.loc[is_internet, 'Medio'] = df.loc[is_internet, 'Medio'].astype(str).str.lower().str.strip().map(internet_map).fillna(df.loc[is_internet, 'Medio'])

    progress_text.info("Paso 4/8: Detectando duplicados con lógica avanzada...")
    
    df['is_duplicate'] = False
    df_reset = df.reset_index().rename(columns={'index': 'original_index'})
    
    df_reset['sort_key'] = df_reset['Título'].str.contains('"', na=False)
    df_reset.sort_values(by=['sort_key', 'Fecha', 'original_index'], ascending=[False, True, True], inplace=True)
    
    rows_list = df_reset.to_dict('records')
    is_duplicate_map = {}

    for i in range(len(rows_list)):
        if rows_list[i]['original_index'] in is_duplicate_map: continue
        for j in range(i + 1, len(rows_list)):
            if rows_list[j]['original_index'] in is_duplicate_map: continue
            
            row1 = rows_list[i]
            row2 = rows_list[j]
            
            if are_duplicates(pd.Series(row1), pd.Series(row2)):
                is_duplicate_map[row2['original_index']] = True
    
    df_reset['is_duplicate'] = df_reset['original_index'].map(is_duplicate_map).fillna(False)
    df = df_reset.sort_values('original_index').set_index('original_index').drop(columns=['sort_key'])

    progress_text.info("Paso 5/8: Aplicando modelos de IA...")
    df_valid = df[~df['is_duplicate']].copy()
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

    progress_text.info("Paso 6/8: Homogeneizando temas...")
    df_valid_homog = df[~df['is_duplicate']].copy()
    if not df_valid_homog.empty and 'Temas Generales - Tema' in df_valid_homog.columns:
        df_valid_homog['titulo_norm_homog'] = df_valid_homog['Título'].apply(normalize_title_for_comparison)
        homogenized_temas = df_valid_homog.groupby('titulo_norm_homog')['Temas Generales - Tema'].transform(lambda x: x.mode()[0] if not x.mode().empty else x)
        df_valid_homog['Temas Generales - Tema'] = homogenized_temas
        df.update(df_valid_homog[['Temas Generales - Tema']])

    progress_text.info("Paso 7/8: Mapeando tema final...")
    if 'Temas Generales - Tema' in df.columns:
        df['Tema'] = df['Temas Generales - Tema'].astype(str).str.strip().map(final_topic_map).fillna('Indefinido')
    
    df.loc[df['is_duplicate'], ['Tono', 'Tema', 'Temas Generales - Tema']] = 'Duplicada'
    
    progress_text.info("Paso 8/8: Generando resultados finales...")
    st.balloons()
    progress_text.success("¡Proceso completado con éxito!")

    final_order = ["ID Noticia", "Fecha", "Hora", "Medio", "Tipo de Medio", "Sección - Programa", "Región", "Título", "Autor - Conductor", "Nro. Pagina", "Dimensión", "Duración - Nro. Caracteres", "CPE", "Tier", "Audiencia", "Tono", "Tema", "Temas Generales - Tema", "Resumen - Aclaracion", "Link Nota", "Link (Streaming - Imagen)", "Menciones - Empresa"]
    df_final = df.copy()

    st.subheader("📊 Resumen del Proceso")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas Totales", len(df_final))
    dups_count = df_final['is_duplicate'].sum()
    col2.metric("Filas Marcadas como Duplicadas", dups_count)
    col3.metric("Filas Únicas", len(df_final) - dups_count)
    
    excel_data = to_excel_from_df(df_final, final_order)
    st.download_button(label="📥 Descargar Archivo Final Procesado", data=excel_data, file_name=f"Dossier_Procesado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.sheet")

    st.subheader("✍️ Previsualización de Resultados")
    final_columns_in_df = [col for col in final_order if col in df_final.columns]
    df_for_editor = df_final[final_columns_in_df].copy()
    if 'Fecha' in df_for_editor.columns:
        df_for_editor['Fecha'] = pd.to_datetime(df_for_editor['Fecha']).dt.strftime('%d/%m/%Y').replace('NaT', '')
    for col_name in ['Link Nota', 'Link (Streaming - Imagen)']:
        if col_name in df_for_editor.columns:
            df_for_editor[col_name] = df_for_editor[col_name].apply(lambda x: 'Link' if pd.notna(x) else '')
    st.dataframe(df_for_editor, use_container_width=True)
    
# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("🚀 Procesador Inteligente de Dossiers v3.6")
st.markdown("Una herramienta para limpiar, enriquecer y analizar dossieres de noticias de forma automática.")
st.info("**Instrucciones:**\n\n1. Prepara tu archivo **Dossier** principal y tu archivo **`Configuracion.xlsx`**.\n2. Sube ambos archivos juntos en el área de abajo.\n3. Haz clic en 'Iniciar Proceso'.")
with st.expander("Ver estructura requerida para `Configuracion.xlsx`"):
    st.markdown("- **`Regiones`**: Columna A (Medio), Columna B (Región).\n- **`Internet`**: Columna A (Medio Original), Columna B (Medio Mapeado).\n- **`Menciones`**: Columna A (Mención Original), Columna B (Mención Mapeada).\n- **`Mapa_Temas`**: Columna A (Temas Generales - Tema), Columna B (Tema).")

uploaded_files = st.file_uploader("Arrastra y suelta tus archivos aquí (Dossier y Configuracion)", type=["xlsx"], accept_multiple_files=True)
dossier_file, config_file = None, None
if uploaded_files:
    for file in uploaded_files:
        if 'config' in file.name.lower(): config_file = file
        else: dossier_file = file
    if dossier_file: st.success(f"Archivo Dossier cargado: **{dossier_file.name}**")
    else: st.warning("No se ha subido un archivo que parezca ser el Dossier.")
    if config_file: st.success(f"Archivo de Configuración cargado: **{config_file.name}**")
    else: st.warning("No se ha subido el archivo `Configuracion.xlsx`.")
if st.button("▶️ Iniciar Proceso Completo", disabled=not (dossier_file and config_file), type="primary"):
    run_full_process(dossier_file, config_file)
