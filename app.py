from flask import Flask, request
from flask import render_template

import pandas as pd

from load import word2vec, fasttext

app = Flask(__name__)


class Main:
    def __init__(self):
        loaders = [word2vec, fasttext]

        self.models = {}
        for loader in loaders:
            model = loader()
            self.models[model[0]] = model[1]

        print(self.models)

    def most_similar(self, word):
        results = {}
        for model in self.models:
            try:
                result = self.models[model].wv.most_similar(word)

                # convert results into an html table
                result = pd.DataFrame(result).to_html()

                results[model] = result
            except Exception as e:
                results[model] = None
                print(e)
        return results


@app.route('/')
def home():
    return render_template('index.html', models=main.models.keys())


@app.route('/search')
def search():
    word = request.args.get('word')

    result = None
    if word is not None:
        result = main.most_similar(word)
    return result


if __name__ == '__main__':
    # load models
    main = Main()

    # run flask server
    app.run(host='0.0.0.0', port='8000')

    
