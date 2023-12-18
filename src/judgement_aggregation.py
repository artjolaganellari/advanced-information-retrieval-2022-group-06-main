## implement part 1 here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import random
from sklearn.metrics import cohen_kappa_score
import math

def getPerfectMatch(data):
    #determine the relevance ratings of all query-document pairs for which all users give the same relevance judgement
    return((data.groupby(
        ["query_id","doc_id"])["userId"].count().reset_index(
            name="uniqueRating").assign(
                uniqueRating = lambda row: row.uniqueRating==1)).merge(
                data.groupby(["query_id","doc_id"])["relevanceLevel"].agg(
                        relevanceLevelPerfectMatch = lambda x: np.max(x) if np.max(x) == np.min(x) else np.nan),
                        on=["query_id","doc_id"], how="inner"))

def getMajority(data):
    #determine the relevance ratings of all query-document pairs for which a single majority vote (mode) exists
    return(data.groupby(["query_id","doc_id"])["relevanceLevel"].agg(
            aggregatedRelevanceLevel = lambda x: list(pd.Series.mode(x))).apply(
                lambda x : x.aggregatedRelevanceLevel[0] if len(x.aggregatedRelevanceLevel)==1 else np.nan,
                axis=1).reset_index(name='relevanceLevelMajority'))

def getHeuristic(data):
    #determine the relevance ratings of all query-document pairs for which multiple majority votes exist (and we have to apply a heuristic and take the best mode)
    return(data.groupby(["query_id","doc_id"])["relevanceLevel"].agg(
            aggregatedRelevanceLevel = lambda x: list(pd.Series.mode(x))).apply(
                lambda x : x.aggregatedRelevanceLevel[math.floor(len(x.aggregatedRelevanceLevel)/2)],
                axis=1).reset_index(name='relevanceLevelHeuristic')
    )

def getRelevanceData(data):
    #determines separate columns with perfect matches, majority votes and heuristics
    dataPerfectMatch = getPerfectMatch(data)
    dataMajority = getMajority(data)
    dataHeuristic = getHeuristic(data)    
    #determine the binary relevance level with heuristics
    dataHeuristicBinary = getHeuristic(data.assign(
        relevanceLevel = 
        np.floor(data.relevanceLevel.str[0].astype("int")/2).astype("int")))    
    dataHeuristicBinary.rename(columns={"relevanceLevelHeuristic": "relevanceLevelHeuristicBinary"},inplace=True)
    return(dataPerfectMatch.merge(
        dataMajority, on=["query_id","doc_id"], how="inner").merge(
            dataHeuristic, on=["query_id","doc_id"], how="inner").merge(
                dataHeuristicBinary, on=["query_id","doc_id"], how="inner"
            ))

def printRelevanceSummary(relevanceResults):
    #prints some summary statistics about the dataframe constructed in getRelevanceData
    nrRatings = relevanceResults.shape[0]
    ratingsUnique = (relevanceResults.uniqueRating.sum())/nrRatings
    ratingsPerfectMatch = (nrRatings-relevanceResults.relevanceLevelPerfectMatch.isna().sum())/nrRatings -ratingsUnique
    ratingsMajority = (nrRatings-relevanceResults.relevanceLevelMajority.isna().sum())/nrRatings -ratingsPerfectMatch - ratingsUnique
    ratingsHeuristic = (nrRatings-relevanceResults.relevanceLevelHeuristic.isna().sum())/nrRatings - ratingsMajority-ratingsPerfectMatch-ratingsUnique
    print("Unique ratings:", ratingsUnique)
    print("Perfect match:", ratingsPerfectMatch)
    print("Majority vote:", ratingsMajority)
    print("Heuristic:", ratingsHeuristic)

def getKappa(data, user_id):
    #determines Cohen's kappa of the ratings of user_id and the ratings of the other users

    #determine the ratings of user_id
    dataUser = data.loc[data.userId == user_id][["query_id", "doc_id", "relevanceLevel"]].rename(columns = {"relevanceLevel":"relevanceLevelUser"})

    #find majority ratings of other users
    dataMajority = (data.loc[
        np.logical_and.reduce((data.userId != user_id , 
        data.query_id.isin(dataUser.query_id),
        data.doc_id.isin(dataUser.doc_id)))
        ].groupby(["query_id","doc_id"])["relevanceLevel"].agg(
            aggregatedRelevanceLevel = lambda x: list(pd.Series.mode(x)))) #aggregate multiple modes to list

    #use heuristic in case of multiple modes: use best relevance judgment
    dataMajority["relevanceLevelMajority"] = (dataMajority.apply(
                lambda x: x.aggregatedRelevanceLevel[math.floor(len(x.aggregatedRelevanceLevel)/2)], #collapse list of modes to "middle" entry
                axis=1)
    )

    #merge relevanceLevels of user with majority relevance levels 
    res = dataUser.merge(dataMajority, on=["query_id","doc_id"], how="inner")    
    
    #if judgement of user_id is among the majority judgements (multiple modes), set the user's judgement as the single majority judgement
    userVotedLikeMajority = [c in l for c, l in zip(res['relevanceLevelUser'], res['aggregatedRelevanceLevel'])] #identify those pairs where user rating corresponds to one of the multiple modes of majority votes    
    res.loc[userVotedLikeMajority,"relevanceLevelMajority"] = res.loc[userVotedLikeMajority,"relevanceLevelUser"]
    
    return(cohen_kappa_score(res.relevanceLevelUser,res.relevanceLevelMajority)) #return cohen's kappa

def excludeLowKappa(data, kappaThreshold=0.2):
    #excludes users from data if their Cohen's Kappa falls below 0.2 (Cohen's kappa calculated in getKappa)
    res = {u: getKappa(data,u) for u in data.userId.drop_duplicates()}
    excludedUsers = [u for u,kappa in res.items() if kappa < kappaThreshold]
    return(data.loc[~ data.userId.isin(excludedUsers)], res)

#set the directory which contains the raw data
dataDir = path.join("..", "data", "Part-1")

baseline = pd.read_csv(path.join(dataDir, "fira-22.baseline-qrels.tsv"), sep=" ", header= None, names =["queryid","hardcoded-Q0", "documentid", "relevance-grade"])

#load the queries and query text
queries = pd.read_csv(path.join(dataDir, "fira-22.queries.tsv"), sep="\t")

#load the documents and document text
documents = pd.read_csv(path.join(dataDir, "fira-22.documents.tsv"), sep="\t")

#load judgements of query-document pairs and process it
judgements = pd.read_csv(path.join(dataDir, "fira-22.judgements-anonymized.tsv"), sep="\t", header= "infer")
judgements['dateTime'] = pd.to_datetime(judgements['judgedAtUnixTS'],unit='s') #convert into dateTime
judgements['durationUsedToJudgeSeconds'] = judgements['durationUsedToJudgeMs']/1000 #convert into Seconds

#get some overview statistics
judgements.describe()
judgements.dtypes

#join the document and query text to the judgements
dataRaw = (queries.merge(
    judgements, 
    left_on="query_id", 
    right_on="queryId", 
    how="inner").merge(
        documents, 
        left_on="documentId", 
        right_on="doc_id")).assign(
    queryDocLen = lambda x: x.query_text.str.len() + x.doc_text.str.len(), #determine the character lengths of the document and queries
    queryLen = lambda x: x.query_text.str.len(),
    docLen = lambda x: x.doc_text.str.len(),
        ) 

#plt.scatter(np.log(dataRaw.queryDocLen),np.log(dataRaw.durationUsedToJudgeSeconds))

#average speed of character reading per minute: ~987 char/min +-118 (SD)
#see https://iovs.arvojournals.org/article.aspx?articleid=2166061 Table 2

avgCharReadingSpeedPerMin = 987
sdCharReadingSpeedPerMin = 118
minRatioReadToBeAbleToJudge = 0.1 #read at least minRatioReadToBeAbleToJudge (ratio) of document to be able to judge the document

dataRaw = dataRaw.assign(minJudgementTimeSeconds = lambda x: (x.queryLen + x.docLen * minRatioReadToBeAbleToJudge)/(avgCharReadingSpeedPerMin+sdCharReadingSpeedPerMin * 3) * 60) #determine minimum judgement time

#exclude all judgements with durationUsedToJudgeSeconds below minJudgementTimeSeconds
data = dataRaw.loc[dataRaw.minJudgementTimeSeconds <= dataRaw.durationUsedToJudgeSeconds]

""" relevanceDataRaw = getRelevanceData(dataRaw)
(relevanceDataRaw.merge(baseline,left_on=["query_id","doc_id"], right_on = ["queryid","documentid"])
    .assign(relevanceLevelHeuristic = lambda row: row.relevanceLevelHeuristic.str[0].astype("int"))
    .query("relevanceLevelHeuristic != `relevance-grade`"))

relevanceData = getRelevanceData(data)

printRelevanceSummary(relevanceData)
printRelevanceSummary(relevanceDataRaw.
        merge(relevanceData[["query_id","doc_id"]], on =["query_id","doc_id"], how="inner")) """


#exclude all judgements with Cohen's Kappa below 0.15
kappaThreshold = 0.15
data, kappaDict = excludeLowKappa(data,kappaThreshold)
excludedUsersLowKappa = [u for u,k in kappaDict.items() if k < kappaThreshold]

#determine the majority vote and heuristic vote for the remaining data
relevanceData = getRelevanceData(data)

#select only relevant  columns
relevanceData = relevanceData.loc[:,["query_id", "doc_id", "relevanceLevelHeuristic", "relevanceLevelHeuristicBinary"]].rename(
    columns = {"relevanceLevelHeuristic": "relevanceLevelGranular", "relevanceLevelHeuristicBinary": "relevanceLevelBinary"})

#convert relevance levels on integer granular scale (i.e. 0-3) and binary scale (0-1)
relevanceData["relevanceLevelGranular"] = relevanceData.relevanceLevelGranular.str[0].astype("int")
#relevanceData["relevanceLevelBinary"] = np.floor(relevanceData.relevanceLevelGranular/2).astype("int")

#store results of relevanceData
relevanceData.to_csv(path.join(dataDir, "qrels_preprocessed.tsv"), sep="\t", index=False)

######################################################################
### in-depth analyses of query-document pairs and resulting judgements
######################################################################

#randomly select 5 query-document pairs of the processed query-document data and judgements
random.seed(123)
rowNrSelected = random.sample(range(relevanceData.shape[0]), 5)
relevanceDataSelected = relevanceData.iloc[rowNrSelected]


#randomly select 5 query-document pairs that also users rated that were excluded due to low Kappa
random.seed(1)
dataExcludedUsersLowKappa = dataRaw.loc[[u in excludedUsersLowKappa for u in dataRaw.userId]][["query_id","doc_id"]].drop_duplicates()
dataExcludedUsersLowKappa = dataExcludedUsersLowKappa.iloc[random.sample(range(dataExcludedUsersLowKappa.shape[0]), 5)] 

relevanceDataSelected = relevanceDataSelected.append(relevanceData.merge(dataExcludedUsersLowKappa, on =["query_id", "doc_id"], how="inner")) #after merging dataExcludedUsersLowKappa with the original data.frame, less than 5 query-document pairs are possible.

#join the selected pairs with the raw (unprocessed) data 
dataRawSelected = dataRaw.merge(relevanceDataSelected, on = ["query_id", "doc_id"], how = "inner")
dataRawSelected.shape

#join the Kappa information to dataRawSelected
kappaDf = pd.DataFrame(kappaDict.values(), index= kappaDict.keys(), columns = ["Cohen's Kappa"]).reset_index().rename(columns={"index":"userId"})
dataRawSelected = dataRawSelected.merge(kappaDf, on =["userId"], how="left")

#assess the results
dataRawSelected.head(dataRawSelected.shape[0])


