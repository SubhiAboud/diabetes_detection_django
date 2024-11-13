from django.shortcuts import render
import pickle

# scaler=pickle.load(open('Diabetes Prediction App\scaler.pkl','rb'))
# lr=pickle.load(open('Diabetes Prediction App\lr.pkl','rb'))


def home(request):
    return render(request, 'index.html')

def getPredictions(age,glucose,bloodPressure,insulin,bmi,skinThickness,diabetesPedigreeFunction):
    model = pickle.load(open('models and data/Pickle/lr.sav','rb'))
    scaled = pickle.load(open('models and data/Pickle/scaler.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [age,glucose,bloodPressure,insulin,bmi,skinThickness,diabetesPedigreeFunction]
    ]))
    
    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    glucose = int(request.GET['glucose'])
    bloodPressure = int(request.GET['bloodPressure'])
    age = int(request.GET['age'])
    insulin = int(request.GET['insulin'])
    bmi = float(request.GET['bmi'])
    skinThickness = int(request.GET['skinThickness'])
    diabetesPedigreeFunction = float(request.GET['diabetesPedigreeFunction'])
    
    
    result = getPredictions(glucose, bloodPressure, age, insulin,
                            bmi, skinThickness, diabetesPedigreeFunction)

    return render(request, 'result.html', {'result': result})
