# dossier_utils.py
import re
import datetime
from openpyxl.styles import Font, Alignment, NamedStyle

# --- Estilos y Funciones Auxiliares ---

def create_custom_style():
    custom_link_style = NamedStyle(name="CustomLink")
    custom_link_style.font = Font(color="0000FF", underline="single")
    custom_link_style.alignment = Alignment(horizontal="left")
    custom_link_style.number_format = '@'
    return custom_link_style

def norm_key(text):
    return re.sub(r'\W+', '', str(text).lower().strip()) if text else ""

def convert_html_entities(text):
    if not isinstance(text, str): return text
    html_entities = {
        'á': 'á', 'é': 'é', 'í': 'í', 'ó': 'ó', 'ú': 'ú', 'ñ': 'ñ',
        'Á': 'Á', 'É': 'É', 'Í': 'Í', 'Ó': 'Ó', 'Ú': 'Ú', 'Ñ': 'Ñ',
        '\"': '\"', '“': '\"', '”': '\"', '‘': "'", '’': "'",
        'Â': '', 'â': '', '€': '', '™': ''
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    return text

def normalize_title(title):
    if not isinstance(title, str): return ""
    title = convert_html_entities(title)
    title = re.sub(r'\s*\|\s*[\w\s]+$', '', title)
    return re.sub(r'\W+', ' ', title).lower().strip()

def corregir_texto(text):
    if not isinstance(text, str): return text
    text = convert_html_entities(text)
    text = re.sub(r'(<br>|\[\.\.\.\]|\s+)', ' ', text).strip()
    match = re.search(r'[A-Z]', text)
    if match: text = text[match.start():]
    if text and not text.endswith('...'): text = text.rstrip('.') + '...'
    return text

def extract_link(cell):
    if cell.hyperlink: return {"value": "Link", "url": cell.hyperlink.target}
    if cell.value and isinstance(cell.value, str):
        match = re.search(r'=HYPERLINK\("([^"]+)"', cell.value)
        if match: return {"value": "Link", "url": match.group(1)}
    return {"value": cell.value, "url": None}

def parse_date(fecha):
    if isinstance(fecha, datetime.datetime): return fecha.date()
    try: return datetime.datetime.strptime(str(fecha).split(" ")[0], "%Y-%m-%d").date()
    except (ValueError, TypeError): return None

def format_date_str(fecha_obj):
    if isinstance(fecha_obj, datetime.date): return fecha_obj.isoformat()
    return str(fecha_obj)[:10]

def es_internet(row, tipo_medio_key_norm):
    return norm_key(row.get(tipo_medio_key_norm)) == 'internet'

def es_radio_o_tv(row, tipo_medio_key_norm):
    tm = norm_key(row.get(tipo_medio_key_norm))
    return tm in {'radio', 'televisión'}

def mark_as_duplicate_to_delete(row, headers_norm_map):
    row['Mantener'] = "Eliminar"
    if headers_norm_map['tono'] in row:
        row[headers_norm_map['tono']] = "Duplicada"
    if headers_norm_map['tema'] in row:
        row[headers_norm_map['tema']] = "-"
    if headers_norm_map['temasgeneralestema'] in row:
        row[headers_norm_map['temasgeneralestema']] = "-"

def is_title_problematic(title):
    if not isinstance(title, str): return False
    if re.search(r'\s*\|\s*[\w\s]+$', title): return True
    if re.search(r'[Ââ€™“”“’‘]', title): return True
    return False

# Reutilizaremos esta función en el app.py
def preprocess_text_for_topic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Evitamos descargar stopwords de nuevo, asumimos que nltk está instalado
    from nltk.corpus import stopwords
    stop_words_list = set(stopwords.words('spanish'))
    token_pattern_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    tokens = token_pattern_re.findall(text.lower())
    return " ".join(tok for tok in tokens if tok not in stop_words_list)
