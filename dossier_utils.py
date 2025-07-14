# dossier_utils.py
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
        'Â': '', 'â': '', '€': '', '™': '', '�': ''
    }
    for entity, char in custom_replacements.items():
        text = text.replace(entity, char)
    return text

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
    # Usa la misma lógica de limpieza que para el output para consistencia
    cleaned_title = clean_title_for_output(title)
    # Reemplaza abreviaturas comunes
    abbreviations = {'tm': 'transporte masivo'}
    for abbr, full_text in abbreviations.items():
        cleaned_title = re.sub(fr'\b{abbr}\b', full_text, cleaned_title, flags=re.IGNORECASE)
    # Normaliza a minúsculas y elimina caracteres no alfanuméricos
    normalized_title = re.sub(r'\W+', ' ', cleaned_title).lower().strip()
    return normalized_title

def corregir_resumen(text: str) -> str:
    """Limpia y formatea el texto del resumen."""
    if not isinstance(text, str):
        return text
    text = convert_html_entities(text)
    # Reemplaza saltos de línea, puntos suspensivos entre corchetes y espacios extra
    text = re.sub(r'(<br\s*/?>|\[\.\.\.\]|\s+)', ' ', text).strip()
    # Empieza el texto desde la primera mayúscula encontrada (si existe)
    match = re.search(r'[A-Z]', text)
    if match:
        text = text[match.start():]
    # Asegura que el texto termine con '...'
    if text and not text.endswith('...'):
        text = text.rstrip('.') + '...'
    return text

def preprocess_text_for_topic(text: str) -> str:
    """Preprocesa texto para el modelo de tópicos (quita stopwords y tokeniza)."""
    if not isinstance(text, str):
        return ""
    # Se asume que NLTK stopwords ya fue descargado por app.py
    stop_words_list = set(stopwords.words('spanish'))
    token_pattern_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    tokens = token_pattern_re.findall(text.lower())
    return " ".join(tok for tok in tokens if tok not in stop_words_list)

# --- Funciones de Excel y DataFrame ---

def extract_link_from_cell(cell):
    """Extrae el target de un hipervínculo de una celda de openpyxl."""
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target
    return cell.value # Devuelve el valor si no hay link

def to_excel_from_df(df: pd.DataFrame, final_order: list) -> bytes:
    """Convierte un DataFrame a un archivo Excel en memoria (bytes)."""
    output = io.BytesIO()
    # Asegura que solo se incluyan las columnas que existen en el DataFrame
    final_columns_in_df = [col for col in final_order if col in df.columns]
    df_to_excel = df[final_columns_in_df]

    with pd.ExcelWriter(
        output,
        engine='xlsxwriter',
        datetime_format='dd/mm/yyyy', # Formato de fecha consistente
        date_format='dd/mm/yyyy'
    ) as writer:
        df_to_excel.to_excel(writer, index=False, sheet_name='Resultado')
        workbook = writer.book
        worksheet = writer.sheets['Resultado']
        link_format = workbook.add_format({'color': 'blue', 'underline': 1})

        # Itera y aplica el formato de hipervínculo
        for col_name in ['Link Nota', 'Link (Streaming - Imagen)']:
            if col_name in df_to_excel.columns:
                col_idx = df_to_excel.columns.get_loc(col_name)
                for row_idx, url in enumerate(df_to_excel[col_name]):
                    # Escribe el URL solo si es un string que parece un link
                    if pd.notna(url) and isinstance(url, str) and url.startswith('http'):
                        worksheet.write_url(row_idx + 1, col_idx, url, link_format, 'Link')
    return output.getvalue()

# --- Funciones de Lógica de Negocio (Duplicados) ---

def calculate_title_quality_score(title: str) -> int:
    """Calcula un puntaje de calidad para un título para ayudar a decidir qué duplicado mantener."""
    if not isinstance(title, str):
        return -999  # Un valor muy bajo para títulos no válidos
    score = 100
    # Penaliza por entidades HTML no decodificadas
    score -= len(re.findall(r'&[#\w]+;', title)) * 10
    # Penaliza por caracteres de reemplazo de errores comunes
    score -= title.count('??') * 5
    score -= title.count('�') * 5
    # Penaliza ligeramente por longitud (títulos más cortos y concisos son a menudo mejores)
    score -= len(title) * 0.01
    return int(score)

def are_duplicates(row1: pd.Series, row2: pd.Series, title_similarity_threshold=0.85, date_proximity_days=1) -> bool:
    """
    Compara dos filas para determinar si son duplicadas basado en Mención, Medio, Título y Fecha.
    Asume que las columnas de Fecha ya son de tipo datetime.
    """
    if row1['Menciones - Empresa'] != row2['Menciones - Empresa']:
        return False
    if row1['Medio'] != row2['Medio']:
        return False

    # Compara títulos normalizados
    titulo1 = normalize_title_for_comparison(row1['Título'])
    titulo2 = normalize_title_for_comparison(row2['Título'])
    
    # Compara fechas (ya deben ser objetos datetime)
    fecha1 = row1['Fecha']
    fecha2 = row2['Fecha']
    if pd.isna(fecha1) or pd.isna(fecha2):
        return False # No se puede comparar si una fecha falta

    # Lógica específica por Tipo de Medio
    if row1['Tipo de Medio'] == 'Internet':
        # Para Internet, no deben tener la misma hora (si está disponible) y las fechas deben ser cercanas
        if 'Hora' in row1 and 'Hora' in row2 and row1['Hora'] == row2['Hora']:
             return False # Probablemente no es un duplicado si las horas son idénticas y registradas
        if abs((fecha1 - fecha2).days) > date_proximity_days:
            return False
        
        # Compara títulos por similitud
        if titulo1 == titulo2 and titulo1 != "": return True
        similarity = SequenceMatcher(None, titulo1, titulo2).ratio()
        if similarity >= title_similarity_threshold: return True
    else: # Prensa, Radio, TV, etc.
        # Para otros medios, la fecha debe ser idéntica
        if fecha1.date() != fecha2.date():
            return False
        # Para Radio/TV, la hora también debe ser idéntica si está presente
        if row1['Tipo de Medio'] in ['Radio', 'Televisión']:
            if 'Hora' in row1 and 'Hora' in row2 and row1['Hora'] != row2['Hora']:
                return False
        
        # Compara títulos
        if titulo1 == titulo2 and titulo1 != "": return True
            
    return False
