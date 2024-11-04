import os
import json
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

simplefilter("ignore")

viridisColorScale = sns.color_palette("viridis")
magmaColorScale = sns.color_palette("magma")
crestColorScale = sns.color_palette("crest")
flareColorScale = sns.color_palette("flare")


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
                print(f"{plotName} Exported Correctly")
            except:
                print("\033[91mExporting the plots wasn't possible, the returned type is not included in the decorator function\033[0m")

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

    print("-------------------- EDA --------------------")

    print("*** UNIQUE COFFEE PRIMARY COLORS ***")
    print(coffee["PrimaryColor"].unique(), "\n") #Exploring the unique color of different types of coffee

    print("*** UNIQUE COFFEE PRIMARY PROCESSING METHODS ***")
    print(coffee["PrimaryProcessingMethod"].unique(), "\n") #Exploring the unique color of different types of coffee

    print("*** UNIQUE REGIONS ***")
    print(coffee["Region"].unique(), "\n")





    print("*** CORRELATION BETWEEN NUMERICAL VARIABLES ***")
    print(coffee.corr(numeric_only=True), "\n") #Checking correlations between the variables























