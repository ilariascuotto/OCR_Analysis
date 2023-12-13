import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import pandas as pd

# Carica i dati dal file JSON
with open('results.json', 'r') as json_file:
    data = json.load(json_file)

# Converte i dati in un DataFrame pandas per una gestione più semplice
df = pd.DataFrame(data['images'])

# Estrai le metriche di similarità per ogni metodo
for method in ['Levenshtein', 'Jaccard', 'Cosine']:
    # Estrai la similarità specifica per il metodo corrente
    df[f'{method}_similarity'] = df['similarities'].apply(lambda x: x[method])

    # Classifica la coppia immagine-metodo
    df[f'{method}_classification'] = df.apply(lambda row: row['classifications'][method], axis=1)

# Ordine significativo delle classificazioni
class_order = ['wrong', 'bad', 'sufficient', 'good', 'excellent']

# Ordina il DataFrame in base all'ordine delle classificazioni
df['classifications'] = pd.Categorical(df['classifications'].apply(lambda x: x['Levenshtein']), categories=class_order, ordered=True)
df = df.sort_values(['classifications'])


# Analisi delle Similarità per Metodo e Immagine
for method in ['Levenshtein', 'Jaccard', 'Cosine']:
    # Calcola statistiche descrittive per ogni metodo
    average_similarity_per_method = df.groupby('method')[f'{method}_similarity'].mean().sort_values()

    plt.figure(figsize=(12, 8))
    sns.barplot(x=average_similarity_per_method.index, y=average_similarity_per_method.values)
    plt.title(f'Analisi delle Similarità Medie per Metodo ({method})')
    plt.xlabel('Metodo')
    plt.ylabel(f'Media Similarità ({method})')
    plt.show()


# Analisi delle Classificazioni
for method in ['Levenshtein', 'Jaccard', 'Cosine']:
    # Conteggio delle classificazioni per ogni metodo
    classifications_count = df.groupby(['method', f'{method}_classification'], observed=False).size().unstack(fill_value=0)

    # Calcola le percentuali invece del numero di immagini
    classifications_percentage = classifications_count.div(classifications_count.sum(axis=1), axis=0) * 100

    # Visualizza un istogramma delle classificazioni per ogni metodo con percentuali
    plt.figure(figsize=(12, 8))
    sns.set_palette("pastel")
    ax = classifications_percentage.plot(kind='bar', stacked=True)

    # Annota le percentuali su ogni barra
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8, color='white')

    plt.title(f'Distribuzione delle Classificazioni per Metodo ({method})')
    plt.xlabel('Metodo')
    plt.legend(title='Classificazione', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yticks([])  # Rimuove le etichette sull'asse y
    plt.show()

# Itera sui metodi di preprocessing
for method in df['method'].unique():
    # Filtra il DataFrame per il metodo corrente
    method_df = df[df['method'] == method]

    # Crea un istogramma con tre colonne per ogni similarità
    plt.figure(figsize=(8, 6))

    # Aggiungi barre per le medie di similarità
    plt.bar('Levenshtein', method_df['Levenshtein_similarity'].mean(), color='blue', label='Levenshtein')
    plt.bar('Jaccard', method_df['Jaccard_similarity'].mean(), color='orange', label='Jaccard')
    plt.bar('Cosine', method_df['Cosine_similarity'].mean(), color='green', label='Cosine')

    plt.title(f'Media delle Similarità per Metodo ({method})')
    plt.ylabel('Media Similarità')
    plt.legend()
    plt.show()



# Crea un istogramma con colonne per ogni metodo e ogni similarità
plt.figure(figsize=(12, 8))
# Itera sui metodi di preprocessing
method_names = df['method'].unique()

# Itera sui metodi di preprocessing
for i, method in enumerate(method_names):
    # Filtra il DataFrame per il metodo corrente
    method_df = df[df['method'] == method]

    # Calcola le medie di similarità
    levenshtein_mean = method_df['Levenshtein_similarity'].mean()
    jaccard_mean = method_df['Jaccard_similarity'].mean()
    cosine_mean = method_df['Cosine_similarity'].mean()

    # Aggiungi barre per le medie di similarità
    plt.bar(i - 0.2, levenshtein_mean, width=0.3, color='blue')
    plt.bar(i, jaccard_mean, width=0.3, color='orange')
    plt.bar(i + 0.2, cosine_mean, width=0.3, color='green')

# Aggiungi una linea a 0.5
plt.axhline(y=0.5, color='red', linestyle='--', label='Linea a 0.5')

# Imposta le etichette sull'asse delle x
plt.xticks(range(len(method_names)), method_names)

# Aggiungi le etichette per Levenshtein, Jaccard e Cosine nella legenda con i rispettivi colori
lev_patch = patches.Patch(color='blue', label='Levenshtein')
jaccard_patch = patches.Patch(color='orange', label='Jaccard')
cosine_patch = patches.Patch(color='green', label='Cosine')
plt.legend(handles=[lev_patch, jaccard_patch, cosine_patch])

plt.title('Medie delle Similarità per Metodo')
plt.xlabel('Metodo')
plt.ylabel('Media Similarità')
plt.show()