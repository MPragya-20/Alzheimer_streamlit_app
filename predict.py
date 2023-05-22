import joblib
 
def predict(data):
    """Loads the model and predicts the class of the instance

    Parameters
    ----------
    data : array
        input data from UI

    Returns
    -------
    list[int]
        returns class - 0 for Demented and 1 for Non-demented
    """
    
    
    nb = joblib.load('nb_model.sav')
    return nb.predict(data) 