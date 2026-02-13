import re
import html
import io
import pandas as pd
from difflib import SequenceMatcher
from nltk.corpus import stopwords

# --- Funciones de limpieza de texto ---

def convert_html_entities(text: str) -> str:
    """Decodifica entidades HTML y reemplaza caracteres problemáticos."""
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    custom_replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        'Â': '', 'â': '', '€': '', '™': '', '': '',
        '\xa0': ' ' 
    }
    for entity, char in custom_replacements.items():
        text = text.replace(entity, char)
    return text

def clean_title(title: str) -> str:
    """
    Limpia SOLO las entidades HTML del título.
    """
    if not isinstance(title, str):
        return ""
    return convert_html_entities(title)

def clean_title_for_output(title: str) -> str:
    """
    Limpia un título para mostrarlo en el resultado final.
    Elimina pipes (|), guiones al final y saltos de línea que ensucian el título.
    """
    if not isinstance(title, str):
        return ""
    title = convert_html_entities(title)
    # Aplanamos saltos de línea para que las regex funcionen bien
    title = title.replace('\n', ' ').replace('\r', ' ')
    
    # Elimina sufijos comunes tipo " | NombreMedio"
    title = re.sub(r'\s+\|.*$', '', title)
    title = re.sub(r'\|\s+.*$', '', title)
    title = re.sub(r'\s+-\s+.*$', '', title) # Guion con espacios rodeándolo
    
    return title.strip()

def normalize_title_for_comparison(title: str) -> str:
    """Normaliza un título para una comparación robusta (minúsculas, sin puntuación)."""
    if not isinstance(title, str):
        return ""
    
    # Usamos la limpieza agresiva (sin pipes, sin enters)
    cleaned_title = clean_title_for_output(title)
    
    abbreviations = {'tm': 'transporte masivo'}
    for abbr, full_text in abbreviations.items():
        cleaned_title = re.sub(fr'\b{abbr}\b', full_text, cleaned_title, flags=re.IGNORECASE)
    
    # Solo caracteres alfanuméricos y minúsculas
    normalized_title = re.sub(r'\W+', ' ', cleaned_title).lower().strip()
    return normalized_title

def corregir_resumen(text: str) -> str:
    """Limpia y formatea el texto del resumen."""
    if not isinstance(text, str):
        return text
    text = convert_html_entities(text)
    text = re.sub(r'(<br\s*/?>|\[\.\.\.\]|\s+)', ' ', text).strip()
    match = re.search(r'[A-Z]', text)
    if match:
        text = text[match.start():]
    if text and not text.endswith('...'):
        text = text.rstrip('.') + '...'
    return text

def preprocess_text_for_topic(text: str) -> str:
    """Preprocesa texto para el modelo de tópicos."""
    if not isinstance(text, str):
        return ""
    try:
        stop_words_list = set(stopwords.words('spanish'))
    except:
        stop_words_list = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo'])
        
    token_pattern_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    tokens = token_pattern_re.findall(text.lower())
    return " ".join(tok for tok in tokens if tok not in stop_words_list)

# --- Funciones de Excel y DataFrame ---

def extract_link_from_cell(cell):
    """Extrae el target de un hipervínculo de una celda."""
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target
    return cell.value

def to_excel_from_df(df: pd.DataFrame, final_order: list) -> bytes:
    """Convierte un DataFrame a bytes de Excel."""
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

# --- Funciones de Lógica de Negocio (Duplicados) ---

def calculate_title_quality_score(title: str) -> int:
    """Calcula calidad del título para priorizar cuál mantener."""
    if not isinstance(title, str):
        return -999
    score = 100
    score -= len(re.findall(r'&[#\w]+;', title)) * 10
    score -= title.count('??') * 5
    score -= title.count('') * 5
    if len(title) > 250: score -= 5
    if len(title) < 15: score -= 20
    # Penalizar títulos con basura visible
    if '\n' in title: score -= 15
    if '|' in title: score -= 5
    return int(score)

def are_duplicates(row1: pd.Series, row2: pd.Series, title_similarity_threshold=0.85, date_proximity_days=1) -> bool:
    """
    Determina si dos filas son duplicadas.
    REGLA CLAVE: 
    - Internet: Permite ventana de días, limpieza de títulos agresiva.
    - Radio/TV: Fecha exacta y HORA debe coincidir (o ser nula). Si Hora es diferente, NO es duplicado.
    """
    titulo1 = normalize_title_for_comparison(row1['Título'])
    titulo2 = normalize_title_for_comparison(row2['Título'])
    
    # Si tras normalizar no queda nada, no comparar
    if not titulo1 or not titulo2:
        return False

    fecha1 = row1['Fecha']
    fecha2 = row2['Fecha']
    
    if pd.isna(fecha1) or pd.isna(fecha2):
        return False

    tipo_medio = row1['Tipo de Medio']

    # --- Lógica Específica por Medio ---
    
    if tipo_medio == 'Internet':
        # Internet: Flexibilidad en fecha
        if abs((fecha1 - fecha2).days) > date_proximity_days:
            return False
            
    elif tipo_medio in ['Radio', 'Televisión']:
        # Radio/TV: Fecha exacta requerida
        if fecha1.date() != fecha2.date():
            return False
            
        # Radio/TV: REGLA DE HORA ESTRICTA
        # Si ambas tienen hora y son diferentes -> Son noticias distintas
        hora1 = row1.get('Hora')
        hora2 = row2.get('Hora')
        
        if pd.notna(hora1) and pd.notna(hora2):
            # Convertir a string para evitar errores de tipo (time vs str)
            if str(hora1).strip() != str(hora2).strip():
                return False
                
    else:
        # Prensa, Revista, etc: Fecha exacta requerida
        if fecha1.date() != fecha2.date():
            return False

    # --- Comparación de Títulos ---

    # 1. Coincidencia exacta post-normalización
    if titulo1 == titulo2:
        return True

    # 2. Contención (Substring) - Para casos "Titulo" vs "Titulo | ACIS"
    # Solo si tienen una longitud decente para evitar falsos positivos
    if len(titulo1) > 15 and len(titulo2) > 15:
        if titulo1 in titulo2 or titulo2 in titulo1:
            return True

    # 3. Similaridad difusa
    similarity = SequenceMatcher(None, titulo1, titulo2).ratio()
    if similarity >= title_similarity_threshold:
        return True
            
    return False

def detect_duplicates_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta duplicados eficientemente agrupando por Medio y Mención."""
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'original_index'})
    df['title_quality'] = df['Título'].apply(calculate_title_quality_score)
    
    # Ordenar por calidad para marcar como "duplicado" la versión "sucia"
    df.sort_values(
        by=['title_quality', 'Fecha', 'original_index'],
        ascending=[False, True, True],
        inplace=True,
        na_position='last'
    )
    
    grouping_keys = ['Medio', 'Menciones - Empresa']
    duplicate_indices = set()
    
    for _, group in df.groupby(grouping_keys, dropna=False):
        if len(group) < 2:
            continue
            
        group_rows = group.to_dict('records')
        
        for i in range(len(group_rows)):
            current = group_rows[i]
            if current['original_index'] in duplicate_indices:
                continue
            
            for j in range(i + 1, len(group_rows)):
                compare = group_rows[j]
                if compare['original_index'] in duplicate_indices:
                    continue

                if are_duplicates(pd.Series(current), pd.Series(compare)):
                    duplicate_indices.add(compare['original_index'])

    df['is_duplicate'] = df['original_index'].isin(duplicate_indices)
    return df.sort_values('original_index').set_index('original_index').drop(columns=['title_quality'])
