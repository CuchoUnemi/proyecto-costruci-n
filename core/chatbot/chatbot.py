import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
from unidecode import unidecode

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Ruta a los archivos de embeddings
EMBEDDINGS_CARRERAS_PATH = 'core/webscraping/scraping_scripts/embeddings_carreras.json'
EMBEDDINGS_INSTITUCIONAL_PATH = 'core/webscraping/scraping_scripts/embeddings_institucional.json'

# Cargar los archivos JSON
script_dir = os.path.dirname(__file__)
scraping_dir = os.path.join(script_dir, '..', 'webscraping', 'scraping_scripts')

# Cargar información de carreras
with open(os.path.join(scraping_dir, 'informacion_carreras.json'), 'r', encoding='utf-8') as f:
    carreras_data = json.load(f)

# Cargar información institucional
with open(os.path.join(scraping_dir, 'informacion_institucional.json'), 'r', encoding='utf-8') as f:
    institucional_data = json.load(f)

class ChatBot:
    def __init__(self, carrera_actual=None):
        self.carrera_actual = carrera_actual

        # Intentar cargar los embeddings desde archivos
        self.embeddings_carreras = self.cargar_embeddings(EMBEDDINGS_CARRERAS_PATH)
        self.embeddings_institucional = self.cargar_embeddings(EMBEDDINGS_INSTITUCIONAL_PATH)

        # Si no se pudieron cargar, generarlos de nuevo
        if not self.embeddings_carreras:
            self.embeddings_carreras = self.generar_embeddings_carreras()
        if not self.embeddings_institucional:
            self.embeddings_institucional = self.generar_embeddings_institucional()

    def cargar_embeddings(self, path):
        """Intenta cargar embeddings desde un archivo JSON."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def guardar_embeddings(self, embeddings, path):
        """Guarda los embeddings en un archivo JSON."""
        with open(path, 'w') as f:
            json.dump(embeddings, f)

    def generar_embeddings(self, texto):
        """Genera embeddings usando OpenAI."""
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def generar_embeddings_carreras(self):
        """Genera embeddings para todas las carreras y los guarda en un archivo."""
        embeddings = []
        for carrera in carreras_data:
            descripcion_completa = f"{carrera.get('nombre', '')} {carrera.get('enlace', '')} {carrera.get('descripcion', '')} {carrera.get('tituloOtorgado', '')} {carrera.get('duracion', '')} {carrera.get('modalidad', '')} {carrera.get('objetivos', '')} {carrera.get('porqueEstudiar', '')} {carrera.get('autoridades', '')} {carrera.get('perfilIngreso', '')} {carrera.get('perfilEgreso', '')} {carrera.get('informacionAdicional', '')}"
            embeddings.append(self.generar_embeddings(descripcion_completa))
        
        # Guardar los embeddings generados en un archivo
        self.guardar_embeddings(embeddings, EMBEDDINGS_CARRERAS_PATH)
        return embeddings

    def generar_embeddings_institucional(self):
        """Genera embeddings para toda la información institucional y los guarda en un archivo."""
        embeddings = []
        for key, value in institucional_data.items():
            embeddings.append(self.generar_embeddings(f"{key} {value}"))
        
        # Guardar los embeddings generados en un archivo
        self.guardar_embeddings(embeddings, EMBEDDINGS_INSTITUCIONAL_PATH)
        return embeddings

    def clasificar_pregunta_con_gpt(self, pregunta):
        """Clasifica la pregunta como 'institucional' o 'carreras' y detecta si menciona una carrera explícita."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Clasifica la siguiente pregunta como 'institucional' o 'carreras'. Responde únicamente con una de estas dos palabras, sin explicaciones adicionales. Si la pregunta es sobre autoridades principales (rector, vicerrector, entre otros), número de estudiantes, facultades, carreras por facultades (las facultades pueden estar escritas en siglas), historia, dirección, cantidad y número de carreras, etc., clasifícala como 'institucional'. Si es sobre la duración de una carrera, autoridades de una carrera (decano, director), título otorgado, perfil de ingreso, modalidades etc., clasifícala como 'carreras'. Pregunta: "},
                    {"role": "user", "content": pregunta},
                ],
            )
            clasificacion = response.choices[0].message.content.strip().lower()
            clasificacion = clasificacion.strip("'").strip('.') 
            print(clasificacion)
            return clasificacion
        except client.error.OpenAIError:
            return "indefinido", None  # Si hay un error en la llamada a la API

    def buscar_respuesta(self, consulta, embeddings, data, es_institucional=False):
        """Busca la mejor respuesta en el dataset usando similitud de coseno."""
        consulta_embedding = self.generar_embeddings(consulta)
        similitudes = cosine_similarity([consulta_embedding], embeddings)[0]
        idx_mejor_respuesta = similitudes.argmax()
        mejor_similitud = similitudes[idx_mejor_respuesta]

        if es_institucional:
            # Para institucional, data es un diccionario
            clave_mejor_respuesta = list(data.keys())[idx_mejor_respuesta]
            return {clave_mejor_respuesta: data[clave_mejor_respuesta]}, mejor_similitud
        else:
            # Para carreras, data es una lista, por lo que accedemos por índice
            return data[idx_mejor_respuesta], mejor_similitud

    def procesar_pregunta(self, pregunta):
        """Primero clasifica la pregunta usando GPT y luego busca en la categoría correcta."""
        tipo_consulta = self.clasificar_pregunta_con_gpt(pregunta)
        print('TIPO DE CONSULTA: ', repr(tipo_consulta))

        if tipo_consulta == "carreras":
            respuesta_carreras, similitud_carreras = self.buscar_respuesta(pregunta, self.embeddings_carreras, carreras_data)
            if respuesta_carreras:
                self.carrera_actual = respuesta_carreras
            return respuesta_carreras, 'carreras'
        elif tipo_consulta == "institucional":
            respuesta_institucional, similitud_institucional = self.buscar_respuesta(pregunta, self.embeddings_institucional, institucional_data, es_institucional=True)
            print('RESPUESTA INSTITUCIONAL: ', respuesta_institucional)
            return respuesta_institucional, 'institucional'
        else:
            return None, None

def actualizar_contador_carrera(carrera_nombre):
    """Actualiza el contador de consultas de una carrera en graphinfo.json."""
    graph_path = 'core/webscraping/scraping_scripts/graphinfo.json'

    # Cargar el archivo `graphinfo.json`
    if os.path.exists(graph_path):
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
    else:
        graph_data = {}

    # Incrementar el contador de la carrera o agregarla si no existe
    if carrera_nombre in graph_data:
        graph_data[carrera_nombre] += 1
    else:
        graph_data[carrera_nombre] = 1

    # Guardar los cambios en `graphinfo.json`
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=4)


def interactuar_con_usuario(message_text, carrera_actual=None):
    try:
        chatbot = ChatBot(carrera_actual=carrera_actual)
        datos_relevantes, tipo_consulta = chatbot.procesar_pregunta(message_text)

        if not datos_relevantes:
            return "Lo siento, no tengo información sobre eso.", None

        # Si la consulta es sobre una carrera, actualizamos el contador en `graphinfo.json`
        if tipo_consulta == 'carreras':
            carrera_nombre = datos_relevantes.get('nombre')  # Suponiendo que `nombre` es la clave del nombre de la carrera
            if carrera_nombre:
                carrera_nombre_sin_tilde = unidecode(carrera_nombre)
                actualizar_contador_carrera(carrera_nombre_sin_tilde)

        if chatbot.carrera_actual:
            nueva_carrera = chatbot.carrera_actual
        else:
            nueva_carrera = None

        contexto = f"La consulta del usuario está relacionada con {tipo_consulta}. Aquí tienes los datos relevantes: {json.dumps(datos_relevantes, ensure_ascii=False, indent=2)}. Genera una respuesta coherente basándote en estos datos."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Responde a la pregunta del usuario basándote estrictamente en los datos proporcionados sobre la universidad estatal de milagro (UNEMI). Limitate a responder solamente sobre la categoría o campo preguntado."},
                {"role": "user", "content": message_text},
                {"role": "assistant", "content": contexto}
            ]
        )

        chatbot_response = response.choices[0].message.content
        return chatbot_response, nueva_carrera

    except client.error.OpenAIError:
        return "Lo siento, he superado mi límite de uso. Por favor, inténtalo más tarde."
    except Exception as e:
        return f"Ocurrió un error: {str(e)}"
