
from CoffeeReading import readDataset, cleanData
from CoffeeAnalysis import EDA, datasetPreprocessing, PrincipalComponents

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")

EDA(coffeeData)

coffeeScaled = datasetPreprocessing(coffeeData)
PrincipalComponents(coffeeScaled)



























