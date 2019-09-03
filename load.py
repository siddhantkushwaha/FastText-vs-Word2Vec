from gensim.models import Word2Vec, FastText


def word2vec():
    path = 'models/word2vec/word2vec.bin'
    model = Word2Vec.load(path)
    return 'Word2Vec', model


def fasttext():
    path = 'models/fasttext/fasttext.bin'
    model = FastText.load(path)
    return 'FastText', model


if __name__ == '__main__':
    ft = fasttext()[1]
    print(ft.wv.most_similar('food'))

    w2v = word2vec()[1]
    print(w2v.wv.most_similar('food'))
