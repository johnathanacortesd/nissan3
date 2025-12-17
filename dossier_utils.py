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
        '"': '"', '"': '"', ''': "'", ''': "'",
        'Â': '', 'â': '', '€': '', '™': '', '�': ''
    }
    for entity, char in custom_replacements.items():
        text = text.replace(entity, char)
    return text

def clean_title(title: str) -> str:
    """
    Limpia SOLO las entidades HTML del título, sin modificar su contenido.
    Preserva guiones, pipes y todo el texto original.
    """
    if not isinstance(title, str):
        return ""
    # Solo decodifica entidades HTML, no elimina nada
    return convert_html_entities(title)

def clean_title_for_output(title: str) -> str:
    """Limpia un título para mostrarlo en el resultado final, eliminando pipes y tags."""
    if not isinstance(title, str):
        return ""
    title = convert_html_entities(title)
    # Elimina todo desde un pipe '|' o un guion '-' hasta el final de la línea
    title = re.sub(r'\s*[|-].*$', '', title).strip()
    return title

def normalize_title_for_comparison(title: str) -> str:
    """Normaliza un título para una comparación robusta (minúsculas, sin puntuación)."""
    if not isinstance(title, str):
        return ""
    cleaned_title = clean_title_for_output(title)
    abbreviations = {'tm': 'transporte masivo'}
    for abbr, full_text in abbreviations.items():
        cleaned_title = re.sub(fr'\b{abbr}\b', full_text, cleaned_title, flags=re.IGNORECASE)
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
    """Preprocesa texto para el modelo de tópicos (quita stopwords y tokeniza)."""
    if not isinstance(text, str):
        return ""
    stop_words_list = set(stopwords.words('spanish'))
    token_pattern_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    tokens = token_pattern_re.findall(text.lower())
    return " ".join(tok for tok in tokens if tok not in stop_words_list)

# --- Funciones de Excel y DataFrame ---

def extract_link_from_cell(cell):
    """Extrae el target de un hipervínculo de una celda de openpyxl."""
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target
    return cell.value

def to_excel_from_df(df: pd.DataFrame, final_order: list) -> bytes:
    """Convierte un DataFrame a un archivo Excel en memoria (bytes)."""
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
    """Calcula un puntaje de calidad para un título para ayudar a decidir qué duplicado mantener."""
    if not isinstance(title, str):
        return -999
    score = 100
    score -= len(re.findall(r'&[#\w]+;', title)) * 10
    score -= title.count('??') * 5
    score -= title.count('�') * 5
    score -= len(title) * 0.01
    return int(score)

def are_duplicates(row1: pd.Series, row2: pd.Series, title_similarity_threshold=0.85, date_proximity_days=1) -> bool:
    """
    Compara dos filas para determinar si son duplicadas. Asume que 'Medio' y 'Mención' ya coinciden.
    """
    titulo1 = normalize_title_for_comparison(row1['Título'])
    titulo2 = normalize_title_for_comparison(row2['Título'])
    
    fecha1 = row1['Fecha']
    fecha2 = row2['Fecha']
    if pd.isna(fecha1) or pd.isna(fecha2):
        return False

    if row1['Tipo de Medio'] == 'Internet':
        if 'Hora' in row1 and 'Hora' in row2 and row1['Hora'] == row2['Hora']:
             return False
        if abs((fecha1 - fecha2).days) > date_proximity_days:
            return False
        
        if titulo1 == titulo2 and titulo1 != "": return True
        similarity = SequenceMatcher(None, titulo1, titulo2).ratio()
        if similarity >= title_similarity_threshold: return True
    else: # Prensa, Radio, TV, etc.
        if fecha1.date() != fecha2.date():
            return False
        if row1['Tipo de Medio'] in ['Radio', 'Televisión']:
            if 'Hora' in row1 and 'Hora' in row2 and row1['Hora'] != row2['Hora']:
                return False
        
        if titulo1 == titulo2 and titulo1 != "": return True
            
    return False

def detect_duplicates_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta duplicados de forma eficiente usando una estrategia de 'groupby'.
    """
    df = df.reset_index().rename(columns={'index': 'original_index'})
    df['title_quality'] = df['Título'].apply(calculate_title_quality_score)
    
    df.sort_values(
        by=['title_quality', 'Fecha', 'original_index'],
        ascending=[False, True, True],
        inplace=True,
        na_position='last'
    )
    
    grouping_keys = ['Medio', 'Menciones - Empresa']
    duplicate_indices = set()
    
    # Itera sobre grupos donde Medio y Mención son iguales
    for _, group in df.groupby(grouping_keys, dropna=False):
        if len(group) < 2:
            continue
            
        group_rows = group.to_dict('records')
        
        # Bucle anidado solo dentro del grupo pequeño
        for i in range(len(group_rows)):
            if group_rows[i]['original_index'] in duplicate_indices:
                continue
            
            for j in range(i + 1, len(group_rows)):
                if group_rows[j]['original_index'] in duplicate_indices:
                    continue

                if are_duplicates(pd.Series(group_rows[i]), pd.Series(group_rows[j])):
                    # La fila 'j' se marca como duplicado porque el df está ordenado por calidad
                    duplicate_indices.add(group_rows[j]['original_index'])

    df['is_duplicate'] = df['original_index'].isin(duplicate_indices)
    
    return df.sort_values('original_index').set_index('original_index').drop(columns=['title_quality'])
