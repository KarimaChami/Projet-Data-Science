import pytest
from Pipline import split_data, Encodage,load_data 

# def test_load_data():
#   assert 
@pytest.fixture 
def data_encoder():
    data = load_data("data")
    data = Encodage(data)
    return data

def test_dimensions_split(data_encoder):
    X_train, X_test, y_train, y_test = split_data(data_encoder, target='Churn')
    # Vérifier la cohérence des dimensions
    assert len(X_train) == len(y_train), "X_train et y_train n'ont pas le même nombre de lignes"
    assert len(X_test) == len(y_test), "X_test et y_test n'ont pas le même nombre de lignes"
    
    print("Dimensions cohérentes après split")

# def test_encodage():
#      data = load_data("data")
#      data_encoder = Encodage(data)
#      col_objet = data_encoder.select_dtypes(include=['object']).columns.tolist()
#      assert len(col_objet)==0,"Il reste des colonnes categorielles non encodées"
#      print("toutes les colonnes catégorielles sont encodées")
