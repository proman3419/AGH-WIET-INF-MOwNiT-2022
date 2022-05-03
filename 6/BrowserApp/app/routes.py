from app import app
from flask import Flask, request, render_template
from app.browser_logic.WikipediaBrowserLogic import WikipediaBrowserLogic


@app.route('/')
def input_form():
    return render_template('input-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['input-field']
    WBL = WikipediaBrowserLogic()
    WBL.fit(_load_dumped=True)
    data = WBL.search_raw(text.split(), 5)
    return render_template('results.html', data=data)


# @app.route('/')
# def input_form():
#     return render_template('input-form.html')


# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['input-field']
#     return text


@app.route('/hello')
def hello():
    return "Hello, World!"
