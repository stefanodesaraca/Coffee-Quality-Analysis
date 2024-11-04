import pandas as pd
import numpy as np
import json
from cleantext import clean
import re
from datetime import datetime

pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)





def readDataset(fileName: str):

    coffee = pd.read_csv(fileName, index_col=False, encoding="utf-8")

    #print(coffee)
    #print("Columns Data Types: \n", coffee.dtypes, "\n")
    #print("General Description of the Dataset: \n", coffee.describe(), "\n")
    #print("Number of Unique Values: \n", coffee.nunique(), "\n")
    #print("Number of NAs: \n", coffee.isna().sum(), "\n")

    return coffee


def convertAltitude(alt: str):

    #print(alt)

    if len(alt) > 4:

        minAlt, maxAlt = re.split(r'\D+', alt)
        minAlt = int(minAlt)
        maxAlt = int(maxAlt)

        approxAlt = np.mean([minAlt, maxAlt])

        return int(approxAlt)

    else:

        return int(alt)


def splitYear(year: str):

    if len(year) > 4:
        year = year[-4:]
    else:
        pass

    if len(year) < 4: raise Exception("Wrong year characters length")

    return year


def convertDateFormat(date: str):

    #print(date)

    date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date)

    date = datetime.strptime(date.strip(), "%B %d, %Y") #Converting the new formatted date into a datetime object

    newDate = date.strftime("%Y-%m-%d")

    #print(newDate)

    return newDate


def cleanBagWeight(bagWeight: str):

    numbers = [c for c in bagWeight if c.isdigit()]
    bagWeight = ''.join(numbers)

    return bagWeight



def cleanData(fileName: str) -> pd.DataFrame:

    coffee = pd.read_csv(fileName, index_col=False, encoding="utf-8")
    coffee = coffee.drop(columns=["Unnamed: 0", "ICO Number", "Certification Body", "Certification Address", "Certification Contact"])
    coffee = coffee.rename(columns={"Country of Origin": "Origin", "Farm Name": "Farm", "Lot Number": "Lot", "Number of Bags": "NBags",
                                    "Bag Weight": "BagWeight", "Altitude": "ApproxAltitude", "In-Country Partner": "Partner", "Harvest Year": "HarvestYear",
                                    "Processing Method": "ProcessingMethod", "Clean Cup": "CleanCup", "Total Cup Points": "Cup Points",
                                    "Moisture Percentage": "Moisture", "Category One Defects": "C1Defects", "Category Two Defects": "C2Defects", "Grading Date": "GradingDate"})

    coffee = coffee.dropna() #Dropping the rows which contain NAs

    coffee["ApproxAltitude"] = coffee["ApproxAltitude"].apply(lambda x: convertAltitude(x)) #Averaging the altitude of the coffee cultivations
    coffee["HarvestYear"] = coffee["HarvestYear"].apply(lambda x: splitYear(x)) #In case the harvest lasted longer than one year we're only interested in knowing the last one

    coffee["Partner"] = coffee["Partner"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))
    coffee["Owner"] = coffee["Owner"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))
    coffee["Producer"] = coffee["Producer"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))
    coffee["Company"] = coffee["Company"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))
    coffee["Mill"] = coffee["Mill"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))
    coffee["Farm"] = coffee["Farm"].apply(lambda x: clean(x, no_punct=True, no_emoji=True, lower=False))

    countryNamesAndCode = pd.read_csv("countryCodesISO3166.csv", index_col=False, keep_default_na=False)
    countryNames = dict(zip(countryNamesAndCode["countryName"], countryNamesAndCode["2LCode"]))

    coffee["ISO3166A2"] = coffee["Origin"].apply(lambda x: countryNames[x]) #Adding the countries' ISO 3166 Alpha-2 codes

    coffee["GradingDate"] = coffee["GradingDate"].apply(lambda x: convertDateFormat(x)) #Converting the various date variations into a standard format
    coffee["Expiration"] = coffee["Expiration"].apply(lambda x: convertDateFormat(x)) #Converting the various date variations into a standard format

    coffee["BagWeight"] = coffee["BagWeight"].apply(lambda x: cleanBagWeight(x)) #Removing the "kg" string from every weight
    coffee["BagWeight"] = coffee["BagWeight"].astype("int32") #Changing the BagWeight column's data type


    coffee["GradingDate"] = pd.to_datetime(coffee["GradingDate"]) #Converting to datetime data type the GradingDate column
    coffee["Expiration"] = pd.to_datetime(coffee["Expiration"]) #Converting to datetime data type the Expiration column


    coffee["ProcessingMethod"] = coffee["ProcessingMethod"].apply(lambda x: re.sub(r"[^\w\s]", ",", x.title())) #Standardizing the separator between the coffee's primary and secondary processing methods
    coffee[["PrimaryProcessingMethod", "SecondaryProcessingMethod"]] = coffee["ProcessingMethod"].str.split(",", expand=True) #Dividing the two processing methods of the coffee (in case there's more than one) into primary and secondary


    coffee["Color"] = coffee["Color"].apply(lambda x: re.sub(r"[^\w\s]", ",", x.title())) #Standardizing the separator between the coffee's primary and secondary colors
    coffee[["PrimaryColor", "SecondaryColor"]] = coffee["Color"].str.split(",", expand=True) #Dividing the two colors of the coffee (in case there's more than one) into primary and secondary

    #The .title() function standardizes the strings which describe the names of the processing methods or the coffee's colors by only capitalizing the first character of the string and lowering the rest

    coffee = coffee.drop(columns=["ProcessingMethod", "Color"]) #Dropping old columns which contained multiple data in them

    coffee[["PrimaryProcessingMethod", "SecondaryProcessingMethod", "PrimaryColor", "SecondaryColor"]].fillna(np.nan) #Just confirming that all NAs will be the Numpy version of it

    coffee["Region"] = coffee["Region"].apply(lambda x: x.title()) #Standardizing the name of the region where the coffee comes from


    return coffee

























