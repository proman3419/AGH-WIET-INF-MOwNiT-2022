from app import app
from flask import Flask, request, render_template
from app.browser_logic.WikipediaBrowserLogic import WikipediaBrowserLogic


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=False)


@app.route('/')
def input_form():
    return render_template('input-form.html')


@app.route('/', methods=['POST'])
def input_form_post():
    query = request.form['query-input-field']
    results_cnt = request.form['resultscnt-input-field']
    search_results = WBL.search_raw(query.split(), int(results_cnt), 
                                    noise_reduction=True, 
                                    noise_reduction_value=0.05)

    return render_template('results.html', query=query, results_cnt=results_cnt,
                           search_results=search_results)
