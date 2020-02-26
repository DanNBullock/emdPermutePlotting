#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:18:55 2020

@author: plab
"""
import os
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance
import numpy as np
import re
import itertools
import operator
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import more_itertools

# test path
#/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd
#test tag
#tractprofile

def extractDataFromBLStruc(projectDir,dataTag):
    dataDirPaths=[]
    projDirContent=os.listdir(projectDir)
    subjectDirNames = [s for s in projDirContent if 'sub-' in s]
    for iSubjectIDs in subjectDirNames:
        curSubjDir=os.path.join(projectDir,iSubjectIDs)
        subjDirContent=os.listdir(curSubjDir)
        dataDir = [s for s in subjDirContent if dataTag in s]
        dataDirPaths.append(os.path.join(curSubjDir,dataDir[0],'profiles'))
    return dataDirPaths



def compileProjectTractProfiles(projectDir,tractName,metricLabel):
    dataTag='tractprofile'
    dataDirPaths=extractDataFromBLStruc(projectDir,dataTag)
    trackProfileFileList=[]
    subjectsList=[]
    


    for iSubjects in dataDirPaths:
        curFileName=tractName + '_profiles.csv' 
        currFilePath=os.path.join(iSubjects,curFileName)
        trackProfileFileList.append(currFilePath)
        
        splitItems=currFilePath.split('/')
        currentMatch = [s for s in splitItems if "sub-" in s]
        #there can be only one
        currentSubjName=currentMatch[0].replace('sub-','')
        subjectsList.append(currentSubjName)
     
        
        outDataFrame=[]
        for iSubjectIndex in range(len(trackProfileFileList)):
            curPath=trackProfileFileList[iSubjectIndex]
            if os.path.isfile(curPath):
                currentTractData=pd.read_csv(curPath)
                currTractDataDim=currentTractData.shape
                currentMetric=np.transpose(pd.Series.to_numpy( currentTractData[metricLabel]))
                splitItems=curPath.split('/')
                currentMatch = [s for s in splitItems if "sub-" in s]
                currentSubjName=currentMatch[0].replace('sub-','')
            
                nodeNames = [0]*currTractDataDim[0]
                for iNodeNames in range(len(nodeNames)):
                    nodeNames[iNodeNames]='node'+str(iNodeNames)
                    
                subjCol=pd.DataFrame([currentSubjName])
                subjCol=subjCol.rename(columns={0:'subjectID'})
                dataRows=pd.DataFrame(currentMetric.reshape(1,currTractDataDim[0]),columns=nodeNames)
                currDataFrame=pd.concat( [subjCol,dataRows],axis=1 )
                
                if not(isinstance(outDataFrame, pd.DataFrame)):
                    outDataFrame=copy.deepcopy(currDataFrame)
                else:
                    outDataFrame=pd.concat([outDataFrame,currDataFrame],axis=0)
           # else:
                #this plots out 
                #print('output not found for subject')
                #print(curPath)
                #loading up a nan vec causes problems
                #currentMetric=np.squeeze(np.empty((1,200)))
               # currentMetric[:]=(np.nan)
     

    return outDataFrame



#this can be generalized to append anything to anything
def appendGroupDataToTable(tractDataFrameIn,keyfilePath):
    #maybe make something conditional up here in case the subject ID column is string based?
    keyFileDataFrame=pd.read_csv(keyfilePath,header=None)
    subjectsCol=tractDataFrameIn['subjectID']
    subjectsColArray=pd.DataFrame.to_numpy(subjectsCol)
    keyfileArray=pd.DataFrame.to_numpy(keyFileDataFrame)


   
    ungroupedIDs=np.setdiff1d(subjectsColArray, [str(i) for i in keyfileArray[:,0]] ,assume_unique=False)
    notPresentIDs=np.setdiff1d( [str(i) for i in keyfileArray[:,0]],subjectsColArray ,assume_unique=False)
    removeRows=np.where(np.isin([str(i) for i in keyfileArray[:,0]],notPresentIDs))
    cleanedKeyfile=np.delete(keyfileArray,removeRows,axis=0)
    ungroupedLabelVec=np.repeat('ungrouped',len(ungroupedIDs))
    
    #switch to string, because why not
    #actually we only care about subjects we have data for.
    cleanedKeyfile[:,0]=[str(i) for i in cleanedKeyfile[:,0]]
    
    ungroupedArray=np.column_stack([ungroupedIDs,ungroupedLabelVec])
    
    outArray=np.vstack([cleanedKeyfile,ungroupedArray])
    
    sortedGroupVec=[0]*len(outArray)
    
    for iSubjects in range(len(outArray)):
        subjectIndex=np.asarray(np.where(subjectsColArray==outArray[iSubjects,0]))[0]
        sortedGroupVec[subjectIndex[0]]=outArray[iSubjects,1]
        

    tractDataFrameOut=copy.deepcopy(tractDataFrameIn)
    print(tractDataFrameOut.shape)
    print(len(sortedGroupVec))
    tractDataFrameOut['group']=sortedGroupVec
    return tractDataFrameOut


def computeTractProfileEMD(tractDataFrameAugmented):
    
    uniqueGroups=np.unique(pd.DataFrame.to_numpy(tractDataFrameAugmented['group']))
    
   
    
    #emd only works for 2 groups?
    #create unique combinations
    groupCombos=["-".join(map(str, comb)) for comb in itertools.combinations(uniqueGroups, 2)]
    groupSize=len(groupCombos)
    frameSize=tractDataFrameAugmented.shape

    outArray=np.zeros([groupSize,frameSize[1]-2])
    
    nodeNamesIndex = [0]*(frameSize[1]-2)
    for iNodeNames in range(len(nodeNamesIndex)):
        nodeNamesIndex[iNodeNames]='node'+str(iNodeNames)
    
    for iComparisons in range(len(groupCombos)):
        #matlab like typing detected
        currComparison=groupCombos[iComparisons]
       
        
        splitGroups=currComparison.split('-')

        currentDistance=np.zeros([1,frameSize[1]-2])
      
        group1Data= np.asarray(tractDataFrameAugmented.loc[tractDataFrameAugmented['group']==splitGroups[0]][nodeNamesIndex])
      
        group2Data= np.asarray(tractDataFrameAugmented.loc[tractDataFrameAugmented['group']==splitGroups[1]][nodeNamesIndex])
      
        for iNodes in range(frameSize[1]-2):
            #develop report output for when a node is ignored
            group1Vals=group1Data[(np.invert(np.isnan(group1Data[:,iNodes]))),iNodes]
            group2Vals=group2Data[(np.invert(np.isnan(group2Data[:,iNodes]))),iNodes]
            
            currentDistance[0,iNodes]=wasserstein_distance(group1Vals,group2Vals)
            
        outArray[iComparisons,:]=currentDistance
    outDataFrame=pd.concat([pd.DataFrame(groupCombos,columns = ['comparison']), pd.DataFrame(outArray,columns = nodeNamesIndex)], axis=1, sort=False)
    return outDataFrame
    
def bootstrapProfileRandomizedEMDdistribution(tractDataFrameAugmented,iterNumber):
    
    #just to get the column headings, I guess
    actualEMDcompute=computeTractProfileEMD(tractDataFrameAugmented)
    groupComparisons=list(actualEMDcompute['comparison']) 
    
    sampleOutSize=actualEMDcompute.shape
    
    tractDataFrameResort=copy.deepcopy(tractDataFrameAugmented)
    dataStack=np.zeros([sampleOutSize[0],sampleOutSize[1],iterNumber])
    dataStackOut=copy.deepcopy(actualEMDcompute)
    dataStackOut['sorting']='actual-EMD'
    columnsLabels=dataStackOut.columns
    nodeLabels= [s for s in columnsLabels if "node" in s]
    nodeColumnIndexes=np.where(np.isin(columnsLabels,nodeLabels))
    
    
    for iIters in range(iterNumber):
        groupAssignment=tractDataFrameResort['group'].tolist()
        #because I want transparancy in the resort
        resortedGroupAssignment=random.sample(groupAssignment, len(groupAssignment))
        #again using copy for transparancy
        tractDataFrameResort['group']=resortedGroupAssignment
        currentShuffleData=computeTractProfileEMD(tractDataFrameResort)
        currentShuffleData['sorting']='reshuffle'
        dataStackOut=pd.concat([dataStackOut, currentShuffleData], axis=0, sort=False, ignore_index=True)
    
    for iComparisons in np.unique(pd.DataFrame.to_numpy(dataStackOut['comparison'])):
        currentComparisonDataFirst=dataStackOut.loc[dataStackOut['comparison']==iComparisons]
        
        currentComparisonDataSecond=np.squeeze(pd.DataFrame.to_numpy(currentComparisonDataFirst.loc[dataStackOut['sorting']=='reshuffle'])[:,nodeColumnIndexes])
        sortedData=np.sort(np.asarray(currentComparisonDataSecond).astype(float),axis=0)
        nnthPercentile=sortedData[int(round(iterNumber-iterNumber*.01)),:]
        middlePerccentile=sortedData[int(round(iterNumber-iterNumber*.5)),:]
        percentileTable=pd.DataFrame(np.hstack([np.vstack([iComparisons,iComparisons]) ,np.vstack([nnthPercentile,middlePerccentile]),np.vstack(['99th-percentile of null model','50th-percentile  of null model'])]),
             columns=columnsLabels)
        dataStackOut=pd.concat([dataStackOut, percentileTable], axis=0, sort=False, ignore_index=True)
        
    return dataStackOut

def emdPlotsFromTractProfiles(projectDir,keyFilePath,tractName,metricLabel,iterNumber):

    saveFigDir=os.path.join(projectDir,'bids','figs')
    if not(os.path.isdir(saveFigDir)):
        os.mkdir(saveFigDir)


    tractDataFrame=compileProjectTractProfiles(projectDir,tractName,metricLabel)

    tractDataFrameAugmented=appendGroupDataToTable(tractDataFrame,keyfilePath)
 
    randEMDDist=bootstrapProfileRandomizedEMDdistribution(tractDataFrameAugmented,iterNumber)

    for iComparisons in np.unique(pd.DataFrame.to_numpy(randEMDDist['comparison'])):
        print(iComparisons)

        EmdSubGroup=randEMDDist.loc[randEMDDist['comparison']==iComparisons]
        EmdSubGroupShuffle=EmdSubGroup.loc[randEMDDist['sorting']=='reshuffle']
        EmdSubGroupOrig=EmdSubGroup.loc[randEMDDist['sorting']!='reshuffle']
        columnNames=EmdSubGroupOrig.columns

        EMDlongFrameShuffle=copy.deepcopy(pd.melt(EmdSubGroupShuffle, id_vars=['comparison','sorting'], value_vars=columnNames[1:201]))
        EMDlongFrameOrig=copy.deepcopy(pd.melt(EmdSubGroupOrig, id_vars=['comparison','sorting'], value_vars=columnNames[1:201]))


        for inodesNameIndexes in range(len(columnNames)-2):
            currentNodeName='node'+str(inodesNameIndexes)
            EMDlongFrameShuffle=EMDlongFrameShuffle.replace({'variable':currentNodeName},inodesNameIndexes)
            EMDlongFrameOrig=EMDlongFrameOrig.replace({'variable':currentNodeName},inodesNameIndexes)
    
        fig, ax = plt.subplots()
        sns.kdeplot(EMDlongFrameShuffle['variable'], EMDlongFrameShuffle['value'],
                 cmap="Purples", shade=True, vertical=True, shade_lowest=False, cbar=True, n_levels=100,ax=ax,cbar_kws={'label': 'Null model density'})
        sns.lineplot(x=EMDlongFrameOrig['variable'], y=EMDlongFrameOrig["value"].astype(float), hue=EMDlongFrameOrig['sorting'],ax=ax)
        ax.set(xlabel='Node Index', ylabel='EMD value')
        fig.suptitle('EMD comparison for '+ iComparisons + ' relative to null for ' + tractName  , fontsize=16)
        figPathName=saveFigDir+'/'+tractName+'_'+iComparisons+'.svg'
        
        fig.savefig(figPathName,dpi=300)
        
        plt.close() 
        
def rankNullSequence(dataStackOut,projectDir,):
    comparisons=np.unique(pd.DataFrame.to_numpy(dataStackOut['comparison']))
    saveDir=os.path.join(projectDir,'bids','figs')
    
    for iComparisons in range(len(comparisons)):
        currentComparison=comparisons[iComparisons]
        #arbitrary paramter for minimum sequence length
        minSeqLength=1
       
        
        #subset corresponding to 
        subset1=dataStackOut.loc[np.logical_and(dataStackOut['comparison']==currentComparison,dataStackOut['sorting']=='reshuffle')]
         #compute the proportion of the total represented by the max ranked model
        maxProportion=1/subset1.shape[0]
        #quick and dirty because exclusive
        queriredProportions=np.arange(-1,int(np.round(np.log10(maxProportion)))-1,-1)
        #why the indexing?
        queriedRanks=np.zeros((1,len(queriredProportions)))[0]
        for iProportions in range(len(queriredProportions)):
            curProp=queriredProportions[iProportions]
            queriedRanks[iProportions]=np.power(10, float(curProp))
        if not(queriedRanks[-1]==maxProportion):
            rankValues=subset1.shape[0]*np.append(queriedRanks,maxProportion)
        else:
            rankValues=subset1.shape[0]*queriedRanks
            
        #inialize table for the current ranking
        #watch the minus two
        comparisonOrderMagnitudeArray=np.zeros((len(rankValues),subset1.shape[1]-2))
        rankFrame=copy.deepcopy(subset1)
        #remember the ordering, we want descending becaues higher emd = greater distance
        for currentRankValIndex in range(len(rankValues)):
            currentRankVal=rankValues[currentRankValIndex]
            print(currentRankVal)
            currBoolMask=pd.DataFrame.to_numpy(rankFrame.rank(axis=0,numeric_only=True,ascending=False))<=currentRankVal
            #30 as an arbitrary limit on the 

            countStruc=np.zeros([currBoolMask.shape[0],200])
            
            for iIterations in range(currBoolMask.shape[0]):
                currentIterTrueLocations=np.where(currBoolMask[iIterations,:])
               
                saved_groups = []
                groupLengths = []
                for group in more_itertools.consecutive_groups(currentIterTrueLocations[0][:]):
                    currentGroup=list(group)
                    groupLengths.append(len(currentGroup))

                       
                    #print(saved_groups[0][-1])
                #print(groupLengths)
           
                for iConsecNodeVals in range(minSeqLength,max(groupLengths,default=2)+1):
                    curNumVec=(np.asarray(groupLengths)-iConsecNodeVals)+1
                    curNumVec[curNumVec<0]=0
                    curMatchesAvailable=np.sum(curNumVec)
                    countStruc[iIterations,iConsecNodeVals]=curMatchesAvailable
                
            comparisonOrderMagnitudeArray[currentRankValIndex,:]=np.sum(countStruc,axis=0)
            MagnitudeArrayDim=comparisonOrderMagnitudeArray.shape
            logMagnitudeArray=np.zeros((MagnitudeArrayDim))
            #always going to be in the row dimension, so dont worry about it
            maxVal=np.max(np.where(comparisonOrderMagnitudeArray))
            for iXvals in range(MagnitudeArrayDim[1]):
                for iYvals in range(MagnitudeArrayDim[0]):
                    if comparisonOrderMagnitudeArray[iYvals,iXvals]>0:
                        logMagnitudeArray[iYvals,iXvals]=np.log10(comparisonOrderMagnitudeArray[iYvals,iXvals])
                        #else do nothing
            columnHeaders=np.hstack(['proportion',range(maxVal)])         
            currentFrame=pd.DataFrame(np.hstack([np.reshape(queriedRanks,(-1,1)),logMagnitudeArray[:,0:maxVal]]),columns=columnHeaders)
            currentFrame=pd.melt(currentFrame,id_vars='proportion')
            currentFrame.columns=['rank proportion threshold','nodes','logNum']
            currentFrame['nodes']=pd.to_numeric(currentFrame["nodes"])
            
            currentFrame=currentFrame.pivot('threshold','nodes','logNum')
            mask = np.zeros_like(currentFrame.values)
            mask[currentFrame.values==0] = True

            fig, ax = plt.subplots()
            
            sns.heatmap(currentFrame,cbar=True,mask=mask,cbar_kws={'label': 'log10 number of contiguous sequences'})
                    
            fig.suptitle('EMD comparison for '+ iComparisons + ' relative to null for ' + tractName  , fontsize=16)
            figPathName=saveDir+'/'+tractName+'_sequenceLengthPlot_'+iComparisons+'.svg'  
            fig.savefig(figPathName,dpi=300)
        
            plt.close()     

        
        
            
        
        
        

newRange= range(len(tractNames))[1:]
for iTracts in newRange:
    currentTract=pd.DataFrame.to_numpy(tractNames.iloc[iTracts])[0]   
    print(currentTract)
    emdPlotsFromTractProfiles(projectDir,keyfilePath,currentTract.replace(' ','_'),metricLabel,5000)    
    
projectDir='/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd'



#/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd/bids/GroupKey_v2.csv
keyfilePath='/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd/bids/GroupKey_v1.csv'

tractNamesPath='/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd/bids/domains.csv'

keyFileDataFrame=pd.read_csv(keyfilePath,header=None)
tractNames=pd.read_csv(tractNamesPath)


#test tract
#leftAslant
tractName='rightAslant'
metricLabel='fa_1'


profileDirs=extractDataFromBLStruc(projectDir,dataTag)

tractDataFrame=compileProjectTractProfiles(projectDir,tractName,metricLabel)

tractDataFrameAugmented=appendGroupDataToTable(tractDataFrame,keyfilePath)
 
randEMDDist=bootstrapProfileRandomizedEMDdistribution(tractDataFrameAugmented,1000)





fig.savefig("/Users/plab/Documents/example.png",dpi=300)