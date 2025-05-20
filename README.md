
 Grid Lock Prediction Using GRU
=================================

This project is focused on predicting traffic congestion at specific urban junctions using Gated Recurrent Unit (GRU) models. The goal is to anticipate gridlock scenarios and assist in optimizing traffic flow using a deep learning approach integrated into a Flask-based web application.

 Key Features
---------------

- Traffic congestion prediction for multiple junctions (J1–J4)
- GRU-based time series modeling
- Clean data ingestion from `traffic.csv`
- Flask-powered web interface for interactive user input
- Visual feedback with result prediction and map overlay

 Project Structure
---------------------

Grid-Lock-Prediction/
├── Grid Lock Prediction_Project/
│   ├── GRU_model_J1.h5, J2.h5, ...     # Trained GRU models
│   ├── traffic.csv                     # Input dataset
│   ├── Model_Building.ipynb            # Model training and evaluation
│   ├── start.py                        # Flask app entry point
│   ├── templates/                      # HTML pages (index, result, welcome)
│   ├── static/images/                  # Background and result images
│   └── flask.ipynb                     # Web app demo notebook


 Dataset
-----------

- `traffic.csv`: Contains historical traffic volume and time-series data for four traffic junctions.
- Data used to train GRU models per junction.

Model Details
-----------------

- Model: GRU (Gated Recurrent Unit)
- Framework: TensorFlow/Keras
- Separate GRU models trained for each junction (J1, J2, J3, J4)
- Evaluated using MSE and time-series prediction visualizations

Web App Preview
--------------------

- `/`: Homepage to input junction data
- `/predict`: Displays predicted congestion output
- Templates include:
  - `index.html`
  - `result.html`
  - `welcome.html`

 Author
---------

**Bhanu Prasad Dharavathu**  
Master’s Student in Artificial Intelligence  
Email: dharavathubahnu340@gmail.com]

 Future Scope
----------------

- Integrate live traffic data from Google Maps or IoT sensors
- Expand to more junctions using federated learning
- Add visualization dashboard with historical vs predicted comparisons

