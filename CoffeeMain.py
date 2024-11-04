from CoffeeReading import readDataset, cleanData
from CoffeeAnalysis import EDA

#coffeeData = readDataset("coffee.csv")

coffeeData = cleanData("coffee.csv")

EDA(coffeeData)




























