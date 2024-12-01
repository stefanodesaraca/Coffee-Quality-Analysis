import os
import json
import inspect
from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import plotly
import plotly.express as px
from functools import wraps
from warnings import simplefilter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis


simplefilter("ignore")

viridisColorScale = sns.color_palette("viridis")
magmaColorScale = sns.color_palette("magma")
crestColorScale = sns.color_palette("crest")
flareColorScale = sns.color_palette("flare")

os.makedirs("ShapiroWilkTests", exist_ok=True)
os.makedirs("EDAPlots", exist_ok=True)
os.makedirs("AnalysisPlots", exist_ok=True)

shapiroWilkPlotsPath = f"./{os.curdir}/ShapiroWilkTests/"
EDAPlotsPath = f"./{os.curdir}/EDAPlots/"
AnalysisPlotsPath = f"./{os.curdir}/AnalysisPlots/"


def savePlots(plotFunction):

    def checkPlots(plotNames, plots):
        if isinstance(plotNames, list) and isinstance(plots, list):
            return True
        else:
            #print("\033[91mCheckPlots: object obtained are not lists\033[0m")
            return False

    def checkPlotsTypeAndSave(plotName, plots, filePath):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(f"{filePath}{plotName}.png", dpi=300)
            print(f"{plotName} Exported Correctly")

        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(f"{filePath}{plotName}.html")
            print(f"{plotName} Exported Correctly")

        else:
            try:
                plt.savefig(f"{filePath}{plotName}.png", dpi=300)
                print(f"{plotName}.png Exported Correctly")
            except:
                print("\033[91mExporting the plots wasn't possible, the returned type is not included between the ones treatable from the decorator function\033[0m")

        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):

        plotsNames, generatedPlots, filePath = plotFunction(*args, **kwargs)
        #print("File path: " + filePath)

        if checkPlots(plotsNames, generatedPlots) is True:

            for plotName, plot in zip(plotsNames, generatedPlots):
                checkPlotsTypeAndSave(plotName, plot, filePath)

        elif checkPlots(plotsNames, generatedPlots) is False:
            #print("Saving Single Graph...")
            checkPlotsTypeAndSave(plotsNames, generatedPlots, filePath)

        else:
            print(f"\033[91mExporting the plots wasn't possible, here's the data types obtained by the decorator: PlotNames: {type(plotsNames)}, Generated Plots (could be a list of plots): {type(generatedPlots)}, File Path: {type(filePath)}\033[0m")

        return None

    return wrapper


def EDA(coffee: pd.DataFrame):

    numericVariablesName = coffee.select_dtypes(include=np.number).columns.tolist()
    numericVariablesName.remove("ID")

    print("-------------------- Short Dataset Overview --------------------")

    print(coffee.head(10))

    print("-------------------- Dataset Info --------------------")
    print("*** COLUMNS ***")
    print(list(coffee.columns), "\n")

    print("*** COLUMN DATA TYPES ***")
    print(coffee.dtypes, "\n\n")

    print("*** NAs BY COLUMN ***")
    print(coffee.isna().sum(), "\n")

    print("*** SHORT OVERVIEW OF THE DATSET ***")
    print(coffee.describe())

    print("-------------------- EDA --------------------")

    print("*** UNIQUE COFFEE PRIMARY COLORS ***")
    print(coffee["PrimaryColor"].unique(), "\n") #Exploring the unique color of different types of coffee

    print("*** UNIQUE COFFEE PRIMARY PROCESSING METHODS ***")
    print(coffee["PrimaryProcessingMethod"].unique(), "\n") #Exploring the unique color of different types of coffee

    print("*** UNIQUE REGIONS ***")
    print(coffee["Region"].unique(), "\n")

    print("*** UNIQUE APPROXIMATE ALTITUDE ***")
    print(coffee["ApproxAltitude"].unique(), "\n")


    coffeeCharacteristics = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'CleanCup', 'Sweetness', 'Overall', 'Defects', 'CupPoints', 'Moisture', 'C1Defects', 'Quakers', 'C2Defects']


    print("############################## COFFEE INFO BY VARIETY ##############################")

    for c in coffeeCharacteristics:

        print(f" ==================================== {c} ====================================".upper())

        medianCharacteristic = coffee[["Variety", f"{c}"]].groupby("Variety", as_index=False).median().rename(columns={f"{c}": f"Median{c}"})
        characteristicSTD = coffee[["Variety", f"{c}"]].groupby("Variety", as_index=False).std(ddof=0).rename(columns={f"{c}": f"STD{c}"}) #Setting the Degrees Of Freedom to 0 to exclude NAs from the standard deviation calculation
        characteristicMinimum = coffee[["Variety", f"{c}"]].groupby("Variety", as_index=False).min().rename(columns={f"{c}": f"Minimum{c}"})
        characteristicMaximum = coffee[["Variety", f"{c}"]].groupby("Variety", as_index=False).max().rename(columns={f"{c}": f"Maximum{c}"})

        coffeeInfo = pd.merge(right=characteristicSTD, left=medianCharacteristic, how="outer", on="Variety")
        coffeeInfo = pd.merge(right=characteristicMinimum, left=coffeeInfo, how="outer", on="Variety")
        coffeeInfo = pd.merge(right=characteristicMaximum, left=coffeeInfo, how="outer", on="Variety")

        print(coffeeInfo, "\n")


    print("Numerical Variables: ", numericVariablesName)

    print("\n*** CORRELATION BETWEEN NUMERICAL VARIABLES ***")
    print(coffee.corr(numeric_only=True), "\n") #Checking correlations between the variables

    @savePlots
    def ShapiroWilkTest(targetFeatureName, data):
        plotName = targetFeatureName + inspect.currentframe().f_code.co_name

        print(f"Shapiro-Wilk Normality Test On \033[92m{targetFeatureName}\033[0m Target Feature")
        _, SWH0PValue = stats.shapiro(data)  # Executing the Shapiro-Wilk Normality Test - This method returns a 'scipy.stats._morestats.ShapiroResult' class object with two parameters inside, the second is the H0 P-Value

        print(f"Normality Probability (H0 Hypothesis P-Value): \033[92m{SWH0PValue}\033[0m")

        fig, ax = plt.subplots()
        SWQQPlot = stats.probplot(data, plot=ax)
        ax.set_title(f"Probability Plot for {targetFeatureName}")


        return plotName, SWQQPlot, shapiroWilkPlotsPath

    for numVarName in numericVariablesName:
        ShapiroWilkTest(numVarName, coffee[numVarName])

    print("\n\n")

    print("*** TOP 10 COFFEES BY AROMA ***")
    print(coffee[["ID", "ApproxAltitude", "Aroma", "Sweetness", "Balance", "PrimaryColor", "Variety", "Origin", "Mill"]].sort_values(by="Aroma", ascending=False).head(10), "\n")

    print("*** TOP 10 COFFEES BY Sweetness ***")
    print(coffee[["ID", "ApproxAltitude", "Aroma", "Sweetness", "Balance", "PrimaryColor", "Variety", "Origin", "Mill"]].sort_values(by="Sweetness", ascending=False).head(10), "\n")

    print("*** TOP 10 COFFEES BY BALANCE ***")
    print(coffee[["ID", "ApproxAltitude", "Aroma", "Sweetness", "Balance", "PrimaryColor", "Variety", "Origin", "Mill"]].sort_values(by="Balance", ascending=False).head(10), "\n")

    print("*** TOP 10 COFFEES BY APPROXIMATE ALTITUDE ***")
    print(coffee[["ID", "ApproxAltitude", "Aroma", "Sweetness", "Balance", "PrimaryColor", "Variety", "Origin", "Mill"]].sort_values(by="ApproxAltitude", ascending=False).head(10), "\n")

    @savePlots
    def distHist(data, targetVariable: str):

        plotName = targetVariable + "Distribution"

        distributionHistogram = sns.displot(data=data, x=targetVariable, kde=True).set(title = f"{targetVariable} Distribution")
        distributionHistogram.tight_layout()

        return plotName, distributionHistogram, EDAPlotsPath


    coffeeNumericalCharacteristics = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Sweetness', 'Overall']

    for varName in coffeeNumericalCharacteristics:
        distHist(coffee[[varName, "PrimaryColor"]], varName)

    return None


def datasetPreprocessing(coffee: pd.DataFrame) -> pd.DataFrame:

    scaler = StandardScaler()

    #Scaling the variables which aren't in the correct order of magnitude compared to the others
    #coffee["CupPoints"] = scaler.fit_transform(coffee["CupPoints"])
    coffee["ApproxAltitude"] = scaler.fit_transform(coffee[["ApproxAltitude"]])

    print("\nScaled data sample:")
    print(coffee.head(10))

    return coffee


def PrincipalComponents(coffee: pd.DataFrame):

    coffee = coffee[[coffee.select_dtypes(include=np.number).columns.tolist()]] #Keeping only numerical columns to execute PCA
    #TODO IMPROVE EFFICIENCY DECLARING THE NEW SCALED AND NUMERICAL COLUMNS ONLY DATAFRAME

    rows, columns = coffee.shape
    print(coffee.shape)

    nComponents = min(rows, columns) #Choosing the number of dimensions based on the lowest number between the rows and columns one

    print("I'm here 1")

    pca = PCA(n_components=nComponents)
    pca.fit(coffee) #Here get calculated all the PCA math: loading scores, the variation each principal component accounts for, etc.
    pca.transform(coffee) #Generation of the coordinates for the PCA plot

    #Generating the PCA scree-plot
    explainedVariancePercentage = np.round(pca.explained_variance_ratio_ * 100, decimals = 1).astype(np.float64) #The .astype(np.float64) is needed because calculation libraries require high precision represented values such as 64bits ones
    #In this case explainedVariancePercentage was going to be of data type "half" which is a 16bits representation, so not precise enough for Python, and thus here it comes the need to solve this problem
    labels = ["PC" + str(x) for x in range(1, len(explainedVariancePercentage)+1)]

    PCAExplainedVarianceResults = zip(explainedVariancePercentage, labels)

    print("I'm here 2")

    print("\n")
    print("Principal Components and Explained Variance: \n")
    print(PCAExplainedVarianceResults)

    print(pca.get_feature_names_out())

    plt.figure(figsize=(16,9))
    plt.bar(x=range(1, len(explainedVariancePercentage)+1), height=explainedVariancePercentage, labels=labels)
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Explained Variance")
    plt.title("PCA Scree Plot")

    plt.savefig(f"{AnalysisPlotsPath}PCAScreePlot.png", dpi=300)



    return












