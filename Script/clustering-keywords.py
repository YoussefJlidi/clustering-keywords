import cohere # On importe la bibliothèque cohere pour utiliser l'API de cohere.ai
import matplotlib.pyplot as plt # On importe la bibliothèque matplotlib pour créer le graphique

# On utilise l'API de cohere.ai pour extraire les embeddings des mots
co = cohere.Client('XXXXXXXXXXXXXXXXXXXX') # On crée une instance de cohere en utilisant notre clé d'API
words = ['marketing','digital','seo','serp','datascience','python','new york','crypto','bitcoin','ethereum'] # On définit les mots pour lesquels on veut les embeddings
response = co.embed(model='large', texts=words) # On utilise la méthode embed de l'API de cohere en spécifiant le modèle 'large' pour les embeddings
embeddings = response.embeddings # On récupère les embeddings dans la variable embeddings

x = [embed[0] for embed in embeddings] # On crée une liste x contenant la première valeur de chaque embedding
y = [embed[1] for embed in embeddings] # On crée une liste y contenant la deuxième valeur de chaque embedding

# On utilise matplotlib pour créer le graphique à bulles
plt.figure(figsize=(20,10)) # On utilise la méthode figure pour augmenter la taille du graphique
plt.scatter(x, y, s=500, c='green') # On utilise la méthode scatter pour créer le graphique à bulles en spécifiant les listes x et y, la taille des bulles (s) et la couleur des bulles (c)
for i, word in enumerate(words): # On utilise la fonction enumerate pour itérer sur les mots
    plt.annotate(word, (x[i], y[i]), fontsize=20) # On utilise la méthode annotate pour ajouter les mots sur le graphique en spécifiant le mot, la position (x[i], y[i]) et la taille de la police
plt.show() # On utilise la méthode show pour afficher le graphique
