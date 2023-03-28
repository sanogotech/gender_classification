
##  scikit-learn

```
pip install  pandas
pip install scikit-learn
```

# gender_classification
A gender classification code in python to predict the gender based on height weight and shoe size

The library used is sklearn or scikit learn

Used four different classifiers for this purpose:

1. Decision Tree Classifier 
2. Random Forest Classifier 
3. Logistic Regression 
4. SVC (Support Vector Classifier)

Also the most accurate of prediction is also calculated using package numpy


##  Execute

Now we can execute the program as

test_data = [[190, 70, 43],[154, 75, 38],[181,65,40]]
test_labels = [‘male’,’female’,’male’]

```
Output:

[‘male’ ‘female’ ‘female’]
[‘male’ ‘female’ ‘female’]
[‘female’ ‘female’ ‘female’]
[‘female’ ‘female’ ‘female’]
```

Decision Tree is the best classifier for this problem

## Processus de validation du modèle de machine 

- 1. Apprentissage du modèle
- 2. Test des modèles (evaluer les modèles) 
- 3. Validation du modèle (choisir le bon modèle) et enrichir la data
- 4. Generer le modele
- 5. Utiliser le modele
