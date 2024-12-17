
from CoffeeReading import readDataset, cleanData, exportCleanData
from CoffeeAnalysis import EDA, datasetPreprocessing, SKLPrincipalComponents, MLXTPrincipalComponents, KMeansClustering

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")
exportCleanData(coffeeData)

EDA(coffeeData)

coffeeScaled = datasetPreprocessing(coffeeData)
SKLPrincipalComponents(coffeeScaled)
MLXTPrincipalComponents(coffeeScaled)

KMeansClustering(coffeeScaled)
























