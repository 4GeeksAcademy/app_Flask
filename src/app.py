from flask import Flask, request, render_template, jsonify
from pickle import load
import traceback
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model with enhanced error handling
try:
    model_path = "../src/decision_tree_regressor_default_42.sav"
    logger.info(f"Attempting to load model from: {model_path}")
    model = load(open(model_path, "rb"))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    
    if request.method == "POST":
           
        try:
            if not model:
                raise Exception("Model not loaded - check server logs")
            
            # Collect and validate form data
            input_data = {
                "Time_spent_Alone": request.form.get("val1"),
                "Stage_fear": request.form.get("val2"),
                "Social_event_attendance": request.form.get("val3"),
                "Going_outside": request.form.get("val4"),
                "Drained_after_socializing": request.form.get("val5"),
                "Friends_circle_size": request.form.get("val6"),
                "Post_frequency": request.form.get("val7"),
                "Personality": request.form.get("val8"),
               
            }            
            
            logger.debug(f"Raw input data: {input_data}")
            
            # Convert to float and handle empty values
            features = []
            for key, value in input_data.items():
                try:
                    features.append(float(value))
                except (ValueError, TypeError) as e:
                    raise Exception(f"Invalid value for {key}: {value}") from e
            
            # Reshape for model prediction
            features_array = np.array(features).reshape(1, -1)
            logger.debug(f"Features array: {features_array}")
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            logger.info(f"Prediction successful: {prediction}")
            #return f"Hola"
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            error = error_msg
    
    return render_template("index.html", 
                         prediction=prediction,
                         error=error)


@app.route('/prediction', methods=['POST'])
def prediction():
    nombre = request.form['val1']
    # Aquí llamás a tu función con los datos del form
    #resultado = mi_funcion(nombre)
    return f"Hola {nombre}, resultado: {14}"

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
