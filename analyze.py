

from flask import request
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask import Flask, render_template




app = Flask(__name__)
app.debug = True



CORS(app)


from Topic_Analyzer import topic_analyze

@app.route('/YouTopics')
def index():


    return render_template('YouTopics.html')




@app.route('/analyze', methods=['POST'])
def get_keywords():
    td = {}
    message = request.get_json(force=True)
    url = message['url']
    topic = message['topic']
    key_string = message['keys']
    keys_clean = key_string.replace(' ','')
    keys = keys_clean.split(",")
    td['{0}'.format(topic)] = keys
    response = topic_analyze(url, td)
    return jsonify(response)

