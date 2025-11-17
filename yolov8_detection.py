from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

# Nom du fichier de l'image de sortie
OUTPUT_IMAGE_NAME = "detected_car.jpg"

def download_sample_image(url="https://ultralytics.com/images/bus.jpg"):
    """Télécharge une image d'exemple pour la détection."""
    print(f"Téléchargement de l'image d'exemple depuis: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Lève une exception pour les codes d'état d'erreur
        img = Image.open(BytesIO(response.content))
        # Sauvegarder l'image localement pour l'inférence
        input_path = "sample_input.jpg"
        img.save(input_path)
        print(f"Image d'entrée sauvegardée sous: {input_path}")
        return input_path
    except Exception as e:
        print(f"Erreur lors du téléchargement ou de l'ouverture de l'image: {e}")
        return None

def run_yolov8_detection(image_path):
    """Charge le modèle YOLOv8 et effectue la détection."""
    if not image_path or not os.path.exists(image_path):
        print("Chemin de l'image invalide ou fichier non trouvé.")
        return

    print("Chargement du modèle YOLOv8n...")
    # Charger un modèle pré-entraîné (YOLOv8n pour nano, rapide et léger)
    model = YOLO('yolov8n.pt')

    print(f"Exécution de la détection sur {image_path}...")
    # Exécuter l'inférence sur l'image
    # save=True va créer un dossier 'runs/detect/predict' et y sauvegarder l'image
    results = model.predict(source=image_path, save=True, conf=0.25)

    # Afficher les résultats (optionnel)
    for r in results:
        print(f"Nombre de boîtes détectées: {len(r.boxes)}")
        # Le chemin de sauvegarde est généralement dans runs/detect/predictX/image_name
        # On peut tenter de trouver le chemin exact si nécessaire, mais save=True est suffisant pour l'exemple.
        
    print("\nDétection terminée. Le résultat est sauvegardé dans le dossier 'runs/detect/predict*'.")
    print("Veuillez vérifier le dossier 'runs' pour l'image de sortie.")


if __name__ == "__main__":
    # Télécharger une image d'exemple (un bus est un bon exemple pour la détection d'objets)
    input_image = download_sample_image()
    
    if input_image:
        run_yolov8_detection(input_image)
    else:
        print("Impossible de continuer sans image d'entrée.")
