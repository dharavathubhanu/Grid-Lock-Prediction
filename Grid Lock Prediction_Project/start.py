from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained model
model1 = load_model('GRU_model_J1.h5')

# Load your pre-trained model
model2 = load_model('GRU_model_J2.h5')

# Load your pre-trained model
model3 = load_model('GRU_model_J3.h5')

# Load your pre-trained model
model4 = load_model('GRU_model_J4.h5')




@app.route('/')
def welcome():
    return render_template('welcome.html')


g_value = -1

@app.route('/road1')
def road1():
    global g_value
    g_value = 1
    return render_template('index.html')

@app.route('/road2')
def road2():
    global g_value
    g_value = 2
    return render_template('index.html')


@app.route('/road3')
def road3():
    global g_value
    g_value = 3
    return render_template('index.html')

@app.route('/road4')
def road4():
    global g_value
    g_value = 4
    return render_template('index.html')


# Define your prediction function
def predict(input_value,g_value):

    # Reshape the input data to match the expected input shape of the model
    # Here, we assume a sequence length of 32
    input_data = np.array([[input_value]] * 32)  # Creating a sequence of length 32 with the same value

    # Reshape the input data to match the expected input shape of the model
    input_data = input_data.reshape(1, 32, 1)  # Reshaping for a single sample


    if(g_value==1):

        # Now, you can pass input_data to your model for prediction
        prediction = model1.predict(input_data)

        rounded_prediction = round(prediction[0][0])
        print(rounded_prediction)

        return rounded_prediction

    if(g_value==2):

        # Now, you can pass input_data to your model for prediction
        prediction = model2.predict(input_data)

        rounded_prediction = round(prediction[0][0])
        print(rounded_prediction)

        return rounded_prediction

    if(g_value==3):

        # Now, you can pass input_data to your model for prediction
        prediction = model3.predict(input_data)

        rounded_prediction = round(prediction[0][0])
        print(rounded_prediction)

        return rounded_prediction

    if(g_value==4):

        # Now, you can pass input_data to your model for prediction
        prediction = model4.predict(input_data)

        rounded_prediction = round(prediction[0][0])
        print(rounded_prediction)

        return rounded_prediction
            

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # Get input value from the form
        input_value = float(request.form['input_value'])

        # Make prediction using the prediction function
        prediction = predict(input_value,g_value)

        if(prediction==0):
            output = "You Can Stay In Your Road, Because it is Free Of Traffic When Compared to other Ways."

            return render_template('result.html', n=output)

        
        else:
            
            # Render the result template with the prediction
            output = "You Can't Stay In this Road, Because it is Full Of Traffic When Compared to other Ways."
            output2 = "So Change the way, Our Suggestion After analysing all the roads is Road Number: "
            return render_template('result.html', n=output,p=output2+str(prediction))

        

if __name__ == '__main__':
    app.run(debug=True)
