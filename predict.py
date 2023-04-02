import joblib
 
def predict(data):
    nb = joblib.load('nb_model.sav')
    return nb.predict(data) 