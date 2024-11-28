from mlxtend.feature_extraction import PrincipalComponentAnalysis

from CoffeeReading import readDataset, cleanData
from CoffeeAnalysis import EDA, datasetPreprocessing

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")

EDA(coffeeData)

coffeeScaled = datasetPreprocessing(coffeeData)
PrincipalComponentAnalysis(coffeeScaled)



























