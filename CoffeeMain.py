
from CoffeeReading import readDataset, cleanData
from CoffeeAnalysis import EDA, datasetPreprocessing, SKLPrincipalComponents, MLXTPrincipalComponents, KMeansClustering

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")

EDA(coffeeData)

coffeeScaled = datasetPreprocessing(coffeeData)
SKLPrincipalComponents(coffeeScaled)
MLXTPrincipalComponents(coffeeScaled)

KMeansClustering(coffeeScaled)
























