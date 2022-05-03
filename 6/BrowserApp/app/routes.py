from app import app
from flask import Flask, request, render_template
from app.browser_logic.Entry import Entry
from app.browser_logic.BrowserLogic import BrowserLogic
from typing import List
import numpy as np
import datasets


def init_entries() -> List[Entry]:
    raw_entries = datasets.load_dataset("wikipedia", "20220301.simple", 
                                        split='train[:2000]')
    entries = []
    for raw_entry in raw_entries:
        entries.append(Entry(raw_entry['url'], raw_entry['title'], 
                                  raw_entry['text'], additional_info={}))
    return np.array(entries)


BL = BrowserLogic('wikipedia', init_entries)
BL.fit(_load_dumped=False)


@app.route('/')
def input_form():
    return render_template('input-form.html')


@app.route('/', methods=['POST'])
def input_form_post():
    query = request.form['query-if']
    results_cnt = int(request.form['resultscnt-if'])
    noise_reduction = True
    try:
        noise_reduction_value = float(request.form['noise-reduction-if'])
    except ValueError:
        noise_reduction = False
        noise_reduction_value = 0
    search_results = BL.search_raw(query.split(), results_cnt, 
                                    noise_reduction=noise_reduction, 
                                    noise_reduction_value=noise_reduction_value)

    return render_template('results.html', query=query, results_cnt=results_cnt,
                           search_results=search_results)
