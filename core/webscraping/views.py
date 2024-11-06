import subprocess
from django.http import JsonResponse
import os
import json
from core.chatbot.chatbot import ChatBot  # Importar la clase ChatBot para usar su lógica de embeddings

# Ruta para guardar los embeddings
EMBEDDINGS_CARRERAS_PATH = 'core/webscraping/scraping_scripts/embeddings_carreras.json'
EMBEDDINGS_INSTITUCIONAL_PATH = 'core/webscraping/scraping_scripts/embeddings_institucional.json'

def actualizar_chatbot(request):
    try:
        # Ejecutar el script de scraping de carreras
        resultado_carreras = subprocess.run(['node', 'core/webscraping/scraping_scripts/scrape_careers.js'], check=True, capture_output=True, text=True)
        print(f"RESULTADO CARRERAS: {resultado_carreras}")
        # Ejecutar el script de scraping de información institucional
        resultado_informacion = subprocess.run(['node', 'core/webscraping/scraping_scripts/scrape_information.js'], check=True, capture_output=True, text=True)
        print(f"RESULTADO INFORMACIÓN: {resultado_informacion}")
        # Crear una instancia del ChatBot para generar embeddings
        chatbot = ChatBot()

        # Generar embeddings de carreras y guardarlos
        embeddings_carreras = chatbot.generar_embeddings_carreras()
        with open(EMBEDDINGS_CARRERAS_PATH, 'w') as f:
            json.dump(embeddings_carreras, f)

        # Generar embeddings de información institucional y guardarlos
        embeddings_institucional = chatbot.generar_embeddings_institucional()
        with open(EMBEDDINGS_INSTITUCIONAL_PATH, 'w') as f:
            json.dump(embeddings_institucional, f)

        return JsonResponse({'message': 'chatbot actualizado y embeddings generados exitosamente'})

    except subprocess.CalledProcessError as e:
        return JsonResponse({'message': 'error al actualizar el chatbot', 'error': str(e)}, status=500)
    except Exception as e:
        return JsonResponse({'message': 'error al generar los embeddings', 'error': str(e)}, status=500)

def graph_info(request):
    # Construir la ruta completa del archivo JSON
    json_path = os.path.join(os.path.dirname(__file__), 'scraping_scripts', 'graphinfo.json')

    # Verificar si el archivo JSON existe; si no, crearlo con un contenido vacío
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump({}, json_file)

    # Cargar los datos del archivo
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Devolver los datos como respuesta JSON
    return JsonResponse(data)