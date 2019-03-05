from flask import Flask, request, jsonify
from flask_cors import CORS
from search import find_film
 

app = Flask(__name__)
CORS(app, resoures={r'/film*': {'origins': '*'}})


@app.route('/film')
def film():
    name = request.args.get('name')# if key doesnt exist, returns none
    print('name parsed = ', name)
    if name == "":
        #print('parsed name is empty')
        return "empty", 200
    result = find_film(name)
    if result is not None:
        print('result is not none and is = \n', result)
        return jsonify(result)
    else:
        print('result = none')
        return "Film not found", 404


if __name__ == '__main__':
    app.run(debug=True)
