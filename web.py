from flask import Flask, render_template, request
import Model as book_recommendation

app = Flask(__name__)

results = []

@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = ""
    results = []
    if request.method == 'POST':
        search_query = request.form['search_query']
        try:
            results = book_recommendation.recommend(search_query)
        except:
            results = []
        print(results)

    return render_template('index.html', results=results, search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True)