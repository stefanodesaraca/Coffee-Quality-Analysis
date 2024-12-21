import os
import json
import inspect

from pandas.core.common import random_state
from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from functools import wraps
from warnings import simplefilter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

simplefilter("ignore")

viridisColorScale = sns.color_palette("viridis")
magmaColorScale = sns.color_palette("magma")
crestColorScale = sns.color_palette("crest")
flareColorScale = sns.color_palette("flare")
pairedColorScale = sns.color_palette("Paired")

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
    numericVariablesName.remove("NBags")
    numericVariablesName.remove("BagWeight")

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


    print("Useful Numerical Variables: ", numericVariablesName)

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


def getNumericalColumnsDataset(data: pd.DataFrame):

    numericColumns = data.select_dtypes(include=np.number).columns.tolist()
    numericColumns.remove("ID")
    numericColumns.remove("NBags")
    numericColumns.remove("BagWeight")

    data = data[numericColumns]  # Overwriting the old dataframe with a new one keeping only numerical columns to then execute PCA. So this is coffee, but only with numerical columns

    return data


def SKLPrincipalComponents(data: pd.DataFrame):

    coffee = getNumericalColumnsDataset(data)

    rows, columns = coffee.shape
    #print("\nDataFrame Shape: ", data.shape)

    nComponents = min(rows, columns)  # Choosing the number of dimensions based on the lowest number between the rows and columns one

    pca = PCA(n_components=nComponents)
    pca.fit(coffee) #Here get calculated all the PCA math: loading scores, the variation each principal component accounts for, etc.
    pca.transform(coffee) #Generation of the coordinates for the PCA plot

    #Generating the PCA scree-plot
    explainedVariancePercentage = np.round(pca.explained_variance_ratio_ * 100, decimals=2).astype(np.float64) #The .astype(np.float64) is needed because calculation libraries require high precision represented values such as 64bits ones
    #In this case explainedVariancePercentage was going to be of data type "half" which is a 16bits representation, so not precise enough for Python, and thus here it comes the need to solve this problem
    labels = ["PC" + str(x) for x in range(1, len(explainedVariancePercentage)+1)]

    SKLPCAExplainedVarianceResults = zip(explainedVariancePercentage, labels)

    print("\n\n*** Scikit-Learn Auto Solver PCA ***")

    print("Principal Components and Explained Variance: ")
    print(list(SKLPCAExplainedVarianceResults))

    explainedVarCumSum = np.cumsum(explainedVariancePercentage)

    plt.figure(figsize=(16, 9))
    bars = plt.bar(x=range(1, len(explainedVariancePercentage)+1), height=explainedVariancePercentage, tick_label=labels, alpha=0.5, label='Explained Variance')
    plt.bar_label(bars, [f"{val:.2f}%" for val in explainedVariancePercentage], padding=3)
    plt.step(range(1, len(explainedVariancePercentage)+1), explainedVarCumSum, where='mid', label='Cumulative Explained Variance')
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance (%)")
    plt.title("Scikit-Learn Auto Solver PCA Scree Plot")
    plt.legend(loc='best')
    plt.grid(axis="y", linestyle="--", linewidth=1.5)

    plt.savefig(f"{AnalysisPlotsPath}PCAScreePlot_SKL.png", dpi=300)


    return None


def MLXTPrincipalComponents(data: pd.DataFrame):

    coffee = getNumericalColumnsDataset(data)

    rows, columns = coffee.shape
    #print("\nDataFrame Shape: ", data.shape)

    nComponents = min(rows, columns)  # Choosing the number of dimensions based on the lowest number between the rows and columns one

    pca = PrincipalComponentAnalysis(n_components=nComponents, solver="svd")
    pca.fit(coffee) #Here get calculated all the PCA math: loading scores, the variation each principal component accounts for, etc.
    pca.transform(coffee)

    explainedVariance = pca.e_vals_normalized_
    explainedVariance = np.round(explainedVariance * 100, decimals=2).astype(np.float64)

    explainedVarCumSum = np.cumsum(pca.e_vals_normalized_*100)


    labels = ["PC" + str(x) for x in range(1, len(explainedVariance)+1)]

    MLXTPCAExplainedVarianceResults = zip(explainedVariance, labels)

    print("\n*** MLXTend SVD Solver PCA ***")

    print("Principal Components and Explained Variance: ")
    print(list(MLXTPCAExplainedVarianceResults))
    print("\n")

    #print("Loading Scores: ")
    #print(pca.loadings_)


    #Generating the PCA scree-plot

    plt.figure(figsize=(16, 9))
    bars = plt.bar(x=range(1, len(explainedVariance)+1), height=explainedVariance, tick_label=labels, alpha=0.5, label='Explained Variance')
    plt.bar_label(bars, [f"{val:.2f}%" for val in explainedVariance], padding=3)
    plt.step(range(1, len(explainedVariance)+1), explainedVarCumSum, where='mid', label='Cumulative Explained Variance')
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance (%)")
    plt.title("MLXTend SVD Solver PCA Scree Plot")
    plt.legend(loc='best')
    plt.grid(axis="y", linestyle="--", linewidth=1.5)


    plt.savefig(f"{AnalysisPlotsPath}PCAScreePlot_MLXT.png", dpi=300)


    return None


def getKMeansClustersFullAnalysis(data: pd.DataFrame, maxK: int):

    #-------------------------- Useful data for later plotting --------------------------

    colVariances = {}

    # Checking every column's variance
    for col in data.columns:
        colVariances.update({col: np.var(data[col])})

    print("\n\nCoffee DataFrame Column Values Variance: ")
    print(colVariances)

    fiftieth = np.percentile(list(colVariances.values()), 50)
    seventyFifth = np.percentile(list(colVariances.values()), 75)
    ninetieth = np.percentile(list(colVariances.values()), 90)
    ninetyFifth = np.percentile(list(colVariances.values()), 95)
    ninetyNinth = np.percentile(list(colVariances.values()), 99)

    print("Columns Variance Distribution Percentiles:")
    print("50th Percentile: ", fiftieth)
    print("75th Percentile: ", seventyFifth)
    print("90th Percentile: ", ninetieth)
    print("95th Percentile: ", ninetyFifth)
    print("99th Percentile: ", ninetyNinth)
    print("\n\n")

    varianceValuableColumns = []

    for c in data.columns:
        if np.var(data[c]) >= seventyFifth: varianceValuableColumns.append(c)  # Only keeping columns which have variance more or equal than the 75th percentile of the distribution made by every column's variance

    # -----------------------------------------------------------------------------------

    means = []
    inertias = []

    for k in range(2, maxK+1):
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    #Manual elbow plot generation
    plt.figure(figsize=(16, 9))
    plt.plot(means, inertias, "o-")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.title("Manual K-Means Elbow Plot")

    plt.savefig(f"{AnalysisPlotsPath}KMeansManualElbowPlot.png", dpi=300)


    #Automatic elbow plot generation using yellowbrick
    km = KMeans(random_state=100)
    visualizer = KElbowVisualizer(km, k=(2, maxK))
    visualizer.fit(data)

    visualizer.show(outpath=f"{AnalysisPlotsPath}KMeansAutomaticElbowPlot.png")


    #Silhouette method

    silhouetteScores = {}
    bestKMetricsAndScores = {}

    for s in range(2, maxK+1):
        sObj = KMeans(n_clusters=s, random_state=100)
        sObj.fit(data)
        labels = sObj.labels_
        #print(labels)

        print(f"\n\nCluster Centroids For K={s}:\n {sObj.cluster_centers_}")

        KMeansClusteringPlot(clusteringData=data, labels=labels, K=s, varianceValuableColumns=varianceValuableColumns)

        silhouetteScoreEuclidean = silhouette_score(data, labels, metric="euclidean", random_state=100)
        silhouetteScoreManhattan = silhouette_score(data, labels, metric="manhattan", random_state=100)
        silhouetteScoreMinkowski = silhouette_score(data, labels, metric="minkowski", random_state=100)

        #print(f"Silhouette score for {s} clusters (Euclidean Distance): ", silhouetteScoreEuclidean)
        #print(f"Silhouette score for {s} clusters (Manhattan Distance): ", silhouetteScoreManhattan)
        #print(f"Silhouette score for {s} clusters (Minkowski Distance): ", silhouetteScoreMinkowski)


        silhouetteScores[s] = {"Euclidean": silhouetteScoreEuclidean,
                               "Manhattan": silhouetteScoreManhattan,
                               "Minkowski": silhouetteScoreMinkowski}

        for scoreDict in silhouetteScores.values():
            bestMetric = None

            bestScore = max(list(scoreDict.values()))

            for key, value in scoreDict.items():
                if scoreDict[key] == bestScore:
                    bestMetric = key

            bestKMetricsAndScores.update({s: {}})
            bestKMetricsAndScores[s].update({bestMetric: bestScore}) #Creating a dictionary which contains the best metric and corresponding score for K clusters

        clusteringRelatedInsights(data, labels)

    labelFeatures = [f"K{i}ClusterLabel" for i in range(2, maxK+1)]
    data.drop(columns=labelFeatures, inplace=True) #Removing cluster label features from the dataframe

    #print(data)

    print("*** ELBOW METHOD ***")

    print("Elbow Values: ", visualizer.elbow_score_)
    print("K Values: ", visualizer.k_values_)
    print("Distance Metric: ", visualizer.distance_metric)
    print("Best K From Elbow Method: ", visualizer.elbow_value_)

    print("\n\n\n*** SILHOUETTE METHOD ***")

    print("All Silhouette Scores For Three Different Metrics For Each K Number of Clusters: ")
    print(silhouetteScores)

    print("\nBest Metric and Score For Each K Number of Clusters:")
    print(bestKMetricsAndScores)


    bestKSilhouette = 0 #The best K
    bestKSilhouetteScore = 0 #The silhouette score of the best K

    for kVal in bestKMetricsAndScores.keys():
        val = bestKMetricsAndScores[kVal]
        #print(val.values())
        val = list(val.values())[0]
        bestKSilhouetteScore = max(bestKSilhouetteScore, val)
        if val >= bestKSilhouetteScore: bestKSilhouette = kVal
        #print(val)
        #print(bestKSilhouette)

    print(f"\nBest K: {bestKSilhouette} | Silhouette Score: {bestKSilhouetteScore}")

    #Executing clustering again to return the best K labels too
    bestKMeans = KMeans(n_clusters=bestKSilhouette, random_state=100)
    bestKMeans.fit(data)
    bestLabels = bestKMeans.labels_

    return bestKSilhouette, bestLabels #Returning the best K obtained from the Silhouette Method since it's more accurate


def KMeansClustering(coffee: pd.DataFrame):

    coffee = getNumericalColumnsDataset(coffee)
    bestK, bestLabels = getKMeansClustersFullAnalysis(coffee, 10)

    return None


@savePlots
def KMeansClusteringPlot(clusteringData: pd.DataFrame, labels: list, K: int, varianceValuableColumns: list):

    clusteringData[f"K{K}ClusterLabel"] = labels  #Adding the Kth KMeans clustering label to each observation

    varianceValuableColumns.append(f"K{K}ClusterLabel") #The Kth clustering labels need to be present in the dataframe by default, otherwise it won't be possible to create the plot in case it doesn't have a variance higher than the 75th percentile of the distribution of columns' variance
    clusteringData = clusteringData[varianceValuableColumns] #A simplified version of the coffee dataframe which only includes columns with variance higher than the 75th of the distribution of every columns' variance

    #print(data.head(10))

    coffeeClustersPlot = sns.PairGrid(clusteringData, hue=f"K{K}ClusterLabel", palette=pairedColorScale)
    coffeeClustersPlot.map_diag(sns.kdeplot)
    coffeeClustersPlot.map_offdiag(sns.scatterplot)
    coffeeClustersPlot.map_lower(sns.kdeplot)
    coffeeClustersPlot.add_legend()
    coffeeClustersPlot.set(title=f"{K} Clusters K-Means Clustering")
    coffeeClustersPlot.tight_layout()

    varianceValuableColumns.remove(f"K{K}ClusterLabel") #Removing the old column which won't be useful for the next plot since it won't contain the right clustering labels anymore
    clusteringData.drop(columns=[f"K{K}ClusterLabel"], inplace=True)

    return f"Coffee{K}ClustersPairPlot", coffeeClustersPlot, AnalysisPlotsPath


#TODO DECORATE WITH SAVEPLOTS, CALL THE FUNCTION IN THE CLUSTERING FOR LOOP
def clusteringRelatedInsights(data: pd.DataFrame, labels: list):

    data["ClusterLabel"] = labels

    threeDVariablesAndClustersViz = px.scatter_3d(data, x='Aroma', y='Acidity', z='Flavor', color='ClusterLabel', color_discrete_map=pairedColorScale, title=f"Aroma, Acidity and Flavor with Markers Colored by Cluster For {max(data["ClusterLabel"])+1} Clusters")

    print("\n\n\n")

    #------------- Cluster-based insights -------------

    insightsVariables = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "CleanCup", "Sweetness"]

    for ins in insightsVariables:

        print(f"{ins}Insights by Cluster for K={max(data["ClusterLabel"])+1}")

        averageXByCluster = data[[ins, "ClusterLabel"]].groupby("ClusterLabel", sort=True, as_index=False, dropna=True).mean()
        print(f"Average {ins} by Cluster:\n", averageXByCluster, "\n")

        stdXByCluster = data[[ins, "ClusterLabel"]].groupby("ClusterLabel", sort=True, as_index=False, dropna=True ).std()
        print(f"{ins} Standard Deviation by Cluster:\n", stdXByCluster, "\n")

        sfpXByCluster = data[[ins, "ClusterLabel"]].groupby("ClusterLabel", sort=True, as_index=False, dropna=True).apply(lambda x: np.percentile(x, 75)).rename(columns={None: f"{ins}75thPercentile"}) #Seventyfifth percentile
        print(f"{ins} 75th Distribution Percentile by Cluster:\n", sfpXByCluster, "\n")

        ntXByCluster = data[[ins, "ClusterLabel"]].groupby("ClusterLabel", sort=True, as_index=False, dropna=True).apply(lambda x: np.percentile(x, 90)).rename(columns={None: f"{ins}90thPercentile"}) #Ninetith percentile
        print(f"{ins} 90th Distribution Percentile by Cluster:\n", ntXByCluster, "\n")


    #TODO BARPLOTS AND SOMETHING ELSE

    #TODO RETURN PLOTS

    return


















































