import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import pandas as pd
import nltk
nltk.download('punkt')


files = [['biology1.txt', 'physics1.txt', 'style1.txt', 'biology2.txt', 'physics2.txt'],
         ['style2.txt', 'biology3.txt', 'physics3.txt', 'style3.txt', 'biology4.txt'],
         ['physics4.txt', 'style4.txt', 'biology5.txt', 'physics5.txt', 'style5.txt']]
num_clusters = 3


def get_file_data(file_name):
    file = open(file_name, 'r')
    title = file.readline()
    data = file.read()
    return title, data


# tokenize_and_stem: tokenizes (splits the texts into a list of its respective words (or tokens)
# and also stems each token)
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    stemmer = SnowballStemmer("russian")
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# tokenize_only: tokenizes the texts only
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def transform_text(texts):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in texts:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'texts', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    return totalvocab_stemmed, totalvocab_tokenized


def vectorization(data):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)

    return tfidf_matrix, terms, dist


def clustering(data, n_clust):
    km = KMeans(n_clusters=n_clust)
    km.fit(data)

    return km


def visualization(clusters, titles, dist):
    MDS()
    mds = MDS(n_components=4, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}

    # set up cluster names using a dict
    cluster_names = {0: 'Biology',
                     1: 'Style',
                     2: 'Physic'}

    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, title=titles))

    groups = df.groupby('label')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Training')
    for name, group in groups:
        ax.plot(group.x, group.y, group.z, marker='o', linestyle='', ms=8,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')

    plt.show()


def print_result(frame, centroids, vocab_frame, terms):
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end=' ')

        for ind in centroids[i, :(num_clusters + 9)]:
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=', ')
        print('\n')

        print("Cluster %d texts:\n" % i, end='')
        for title in frame.loc[i]['title'].values.tolist():
            print(' %s' % title, end='')
        print('\n\n')


def print_predicted_result(frame):
    print("Predicted texts:\n")
    for i in range(num_clusters):
        print("Cluster %d predicted texts:\n" % i, end='')
        for title in frame.loc[i]['title'].values.tolist():
            print(' %s' % title, end='')
        print('\n\n')


def test_model(train, test, kmodel):
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    vect = vectorizer.fit(train)
    vect_data = vect.transform(test)
    predictions = kmodel.predict(vect_data)
    dist = 1 - cosine_similarity(vect_data)
    return predictions, dist


def main():
    # Prepare data for creating model
    titles = []
    texts = []
    for i in files:
        for text in i:
            title, text_data = get_file_data('files/' + text)
            titles.append(title)
            texts.append(text_data)

    test_titles = titles[9:]
    test_texts = texts[9:]
    train_texts = texts[:9]
    train_titles = titles[:9]

    # Model`s training
    train_totalvocab_stemmed, train_totalvocab_tokenized = transform_text(train_texts)
    train_vocab_frame = pd.DataFrame({'words': train_totalvocab_tokenized}, index=train_totalvocab_stemmed)

    train_vect_data, train_terms, train_dist = vectorization(train_texts)  # Create text`s vectors

    # K-means clustering
    km = clustering(train_vect_data, num_clusters)
    clusters = km.labels_.tolist()
    centroids = km.cluster_centers_.argsort()[:, ::-1]

    # Pandas dataframe with results of clustering
    train_ptexts = {'title': train_titles, 'cluster': clusters}
    train_frame = pd.DataFrame(train_ptexts, index=[clusters], columns=['title', 'cluster'])

    # Show results
    print_result(train_frame, centroids, train_vocab_frame, train_terms)
    visualization(clusters, train_titles, train_dist)

    # Check model
    predicted_clusters, predicted_dist = test_model(train_texts, test_texts, km)

    # Pandas dataframes with results of checking
    test_ptexts = {'title': test_titles, 'cluster': predicted_clusters}
    test_frame = pd.DataFrame(test_ptexts, index=[predicted_clusters], columns=['title', 'cluster'])
    print_predicted_result(test_frame)


if __name__ == '__main__':
    main()
