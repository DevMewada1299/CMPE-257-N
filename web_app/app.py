from flask import Flask, render_template, request
import prediction
from datetime import datetime, timedelta

import prediction.current_predictions

app = Flask(__name__)

# Function to adjust for Mondays (use Friday's data)


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = {}
    selected_date = None

    if request.method == "POST":
        date_input = request.form["date"]
        try:
            # Parse date input
            selected_date = date_input
            
            
            # Call prediction model
            predictions = prediction.current_predictions.getPredsOnCurrentDay(selected_date)

        except ValueError:
            return "Noooooo"

    return render_template("results.html", predictions=predictions, selected_date=selected_date)

if __name__ == "__main__":
    app.run(debug=True)