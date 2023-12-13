import cv2
import pytesseract
import Levenshtein
import json
from nltk.metrics import jaccard_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lista dei percorsi delle immagini
image_paths = [f'img2/{str(var)}.jpg' for var in range(1, 101)]

# Lista dei percorsi dei file di testo
text_paths = [f'txt/{str(var)}.txt' for var in range(1, 101)]

# Funzione per calcolare la similarità tra due stringhe utilizzando diversi metodi
def calculate_similarity(text1, text2, methods=['Levenshtein', 'Jaccard', 'Cosine']):
    similarities = {}
    for method in methods:
        if method == 'Levenshtein':
            similarities[method] = Levenshtein.ratio(text1, text2)
        elif method == 'Jaccard':
            set1 = set(text1.split())
            set2 = set(text2.split())
            similarities[method] = 1 - jaccard_distance(set1, set2)
        elif method == 'Cosine':
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
            similarities[method] = cosine_sim
        else:
            raise ValueError(f"Metodo di similarità non supportato: {method}")
    return similarities

def classify_similarity(similarity_scores, class_labels):
    classifications = {}

    for method, score in similarity_scores.items():
        classification = class_labels[-1]  # Imposta l'etichetta più alta di default

        if score < 0.5:
            for label in class_labels:
                if score < 0.5:
                    classification = label
                    break

        classifications[method] = classification

    return classifications

def analyze_image(image):
    if image is None:
        print("Errore nel caricamento dell'immagine.")
    else:
        # Verifica il numero di canali dell'immagine
        num_channels = image.shape[2] if len(image.shape) == 3 else 1
        # Se l'immagine ha più di un canale, convertila in scala di grigi
        if num_channels > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # L'immagine è già in scala di grigi
        # Utilizza Tesseract per estrarre il testo e i dettagli delle parole
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(gray, config=custom_config)


# Crea un dizionario per conservare i risultati
results = {
    "images": []
}

# Per ogni immagine
for i in range(len(image_paths)):
    img_path = image_paths[i]
    text_path = text_paths[i]
    
    print(f'Percorso del file immagine: {img_path}') 
    print(f'Percorso del file di testo: {text_path}')
    
    # Carica l'immagine originale
    original_image = cv2.imread(img_path)
    
    # Estrai il testo manuale associato all'immagine
    with open(text_path, 'r') as file:
        manual_extracted_text = file.read()
    


    # Etichette per le classi
    class_labels = ['wrong', 'good']

    # Definisci le tecniche di preelaborazione
    methods = ['Original', 'Grayscale', 'Canny', 'Binary', 'Equalized', 'Denoised', 'Segmented']

    # Dichiarazione delle liste per questa immagine
    similarity_scores = {}
    classifications = {}
    preprocessed_images = []
    
    for method in methods: 
        # Applica il metodo di preelaborazione
        if method == 'Original':
            preprocessed_image = original_image.copy()  # Crea una copia dell'immagine originale
        elif method == 'Grayscale':
            preprocessed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        elif method == 'Canny':
            preprocessed_image = cv2.Canny(original_image, 100, 200)
        elif method == 'Binary':
            _, preprocessed_image = cv2.threshold(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
        elif method == 'Equalized':
            preprocessed_image = cv2.equalizeHist(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY))
        elif method == 'Denoised':
            preprocessed_image = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 15)
        elif method == 'Segmented':
            _, preprocessed_image = cv2.threshold(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Estrai il testo utilizzando il metodo di preelaborazione
        extracted_text = analyze_image(preprocessed_image)
        
        # Calcola le similarità
        similarity_scores[method] = calculate_similarity(manual_extracted_text, extracted_text)

        # Classifica la coppia immagine-metodo
        classifications[method] = classify_similarity(similarity_scores[method], class_labels)

        # Aggiungi le informazioni al dizionario dei risultati
        image_result = {
            "image_path": img_path,
            "method": method,
            "similarities": similarity_scores[method],
            "classifications": classifications[method],
        }
        results["images"].append(image_result)

        print(f"Similarità per {method}: {similarity_scores[method]}")
        print(f"Classificazione per {method}: {classifications[method]}")

# Scrivi i risultati nel file JSON
with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Risultati scritti nel file results.json")
