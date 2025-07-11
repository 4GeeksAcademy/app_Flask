from flask import Flask, request, render_template
from pickle import load
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir,"./models/random_forest_classifier_default_42.sav")
model = load(open(model_path, "rb"))


class_dict = {
    "0": "Introvertido",
    "1": "Extrovertido",

}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        val5 = float(request.form["val5"])
        val6 = float(request.form["val6"])
        val7 = float(request.form["val7"])
        
        data = [[val1, val2, val3, val4, val5, val6, val7]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)