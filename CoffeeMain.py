
from CoffeeReading import readDataset, cleanData
from CoffeeAnalysis import EDA, datasetPreprocessing, SKLPrincipalComponents, MLXTPrincipalComponents

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")

EDA(coffeeData)

coffeeScaled = datasetPreprocessing(coffeeData)
SKLPrincipalComponents(coffeeScaled)
MLXTPrincipalComponents(coffeeScaled)


























