import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def recommend(title, cosine_sim):
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]
    return movies_cleaned['original_title'].iloc[movie_indices]
def recommend_kmeans(title):
    cluster = movies_cleaned.loc[movies_cleaned['original_title'] == title]['cluster'].iloc[0]
    return movies_cleaned[movies_cleaned['cluster'] == cluster]['original_title'].tolist()

# Carregar e pré-processar datasets
credit_dataset = pd.read_csv("./datasets/tmdb_5000_credits.csv")
movies_dataset = pd.read_csv("./datasets/tmdb_5000_movies.csv")

movies_cleaned = movies_dataset.merge(credit_dataset.rename(columns={"movie_id": "id"}), on='id')
movies_cleaned = movies_cleaned.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_cleaned = movies_cleaned.dropna(subset=['overview'])

# Transformar descrições em matriz TF-IDF
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])





##Search the number of different types of clients
wcss = []
range_values = range(1, 100)  # Aqui, estamos considerando de 1 a 20 clusters, mas você pode ajustar conforme necessário.

for k in range_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(tfv_matrix)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(range_values, wcss, marker='o', linestyle='--')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

print("Using K-Means")
# Definir o número de clusters. Este é um hiperparâmetro que você pode ajustar.
num_clusters = 100

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tfv_matrix)
movies_cleaned['cluster'] = kmeans.labels_
recommendations = recommend_kmeans('The Matrix')
print(recommendations)
##Como visto aqui, o Kmeans não é o método mais aconselhado, devido a grande quantidade de possíveis clusters.


print("Using cosine similaty")
# Calcular similaridade do cosseno
cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

# Criar índice para títulos de filmes
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
# Obter recomendações para "Pulse"
recommendations = recommend('Pulse', cosine_sim)

print(recommendations)



