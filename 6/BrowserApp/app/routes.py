from app import app
from flask import Flask, request, render_template
from app.browser_logic.WikipediaBrowserLogic import WikipediaBrowserLogic


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=True)


@app.route('/')
def input_form():
    return render_template('input-form.html')


@app.route('/', methods=['POST'])
def input_form_post():
    query = request.form['query-if']
    results_cnt = int(request.form['resultscnt-if'])
    noise_reduction = bool(request.form['noise-reduction-cb'])
    noise_reduction_value = float(request.form['noise-reduction-if'])
    search_results = WBL.search_raw(query.split(), results_cnt, 
                                    noise_reduction=noise_reduction, 
                                    noise_reduction_value=noise_reduction_value)

    return render_template('results.html', query=query, results_cnt=results_cnt,
                           search_results=search_results)
