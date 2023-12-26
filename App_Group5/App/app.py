from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


app = Flask(__name__)

# Tải mô hình từ file joblib

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ frontend
        print("hello")
        age = int(request.form['Age'])
        distance = int(request.form['Distance'])
        wifi = int(request.form['Wifi'])
        onlBooking = int(request.form['onlBooking'])
        food = int(request.form['Food'])
        boarding = int(request.form['boarding'])
        seat = int(request.form['seat'])
        entertainment = int(request.form['entertainment'])
        onboard = int(request.form['onboard'])
        legroom = int(request.form['legroom'])
        baggage = int(request.form['baggage'])
        checkin = int(request.form['checkin'])
        inflight_ser = int(request.form['inflight_ser'])
        cleanliness = int(request.form['cleanliness'])
        delay = int(request.form['delay'])

        # Class ticket
        class_ticket = request.form['classTicket']
        if class_ticket == "Business":
            Class_Business = 1
        else:
            Class_Business = 0

        if class_ticket == "Eco":
            Class_Eco = 1
        else:
            Class_Eco = 0

        if class_ticket == "EcoPlus":
            Class_Eco_Plus = 1
        else:
            Class_Eco_Plus = 0

        #Loyal Customer
        loyal_customer = request.form['Loyalcustomer']
        if loyal_customer == "Loyal":
            loyal = 1
        else:
            loyal = 0

        if loyal_customer == "Disloyal":
            disloyal = 1
        else:
            disloyal = 0

        #Type of travel
        Type_travel = request.form['TypeTravel']
        if Type_travel == "Personal":
            personal = 1
        else:
            personal = 0

        if Type_travel == "Business":
            bussiness = 1
        else:
            bussiness = 0

        # Dự đoán với mô hình

        header = ['Age', 'Flight Distance', 'Inflight wifi service',
          'Ease of Online booking', 'Food and drink', 'Online boarding',
          'Seat comfort', 'Inflight entertainment', 'On-board service',
          'Leg room service', 'Baggage handling', 'Checkin service',
          'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
          'Customer Type_Loyal Customer', 'Customer Type_disloyal Customer',
          'Type of Travel_Business travel', 'Type of Travel_Personal Travel',
          'Class_Business', 'Class_Eco', 'Class_Eco Plus']
        #nhận kết quả tại đây
        arr = [age, distance, wifi, onlBooking, food, boarding, seat, entertainment, 
                                onboard, legroom, baggage, checkin, inflight_ser, cleanliness, delay, loyal, 
                                disloyal, bussiness, personal, Class_Business, Class_Eco, Class_Eco_Plus]
        print(arr)
        df = pd.DataFrame(data=[arr], columns=header)

        #Các cột sẽ chuẩn hóa
        columns_to_scale = ['Age', 'Flight Distance', 'Inflight wifi service',
            'Ease of Online booking', 'Food and drink', 'Online boarding',
            'Seat comfort', 'Inflight entertainment', 'On-board service',
            'Leg room service', 'Baggage handling', 'Checkin service',
            'Inflight service', 'Cleanliness', 'Departure Delay in Minutes']

        #Chuẩn hóa data
        df[columns_to_scale] = df[columns_to_scale].astype(int)
        # Kiểm tra lại kiểu dữ liệu và chuẩn hóa
        scaler = joblib.load('scaler.joblib')
        scaled_values = scaler.transform(df[columns_to_scale])
        df[columns_to_scale] = scaled_values

        print(df)

        model = joblib.load('KNN.joblib')
        prediction = model.predict(df)
        print('Kết quả là: ', prediction)

        # # Trả kết quả về frontend
        # return jsonify({'prediction': int(prediction)})

        # Gọi hàm JavaScript để hiển thị thông báo
        return render_template('index.html', prediction=int(prediction))

    except Exception as e:
        print(e)
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)