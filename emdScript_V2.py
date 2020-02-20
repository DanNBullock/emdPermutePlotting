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
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns

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
            else:
                print('output not found for subject')
                print(curPath)
                #loading up a nan vec causes problems
                #currentMetric=np.squeeze(np.empty((1,200)))
               # currentMetric[:]=(np.nan)
     

    return outDataFrame



#this can be generalized to append anything to anything
def appendGroupDataToTable(tractDataFrameIn,keyfilePath):
    #maybe make something conditional up here in case the subject ID column is string based?
    keyFileDataFrame=pd.read_csv(keyfilePath,header=None)
    subjectsCol=tractDataFrame['subjectID']
    subjectsColArray=pd.DataFrame.to_numpy(subjectsCol)
    keyfileArray=pd.DataFrame.to_numpy(keyFileDataFrame)
   
    ungroupedIDs=np.setdiff1d(subjectsColArray, [str(i) for i in keyfileArray[:,0]] ,assume_unique=False)
    notPresentIDs=np.setdiff1d( [str(i) for i in keyfileArray[:,0]],subjectsColArray ,assume_unique=False)
    removeRows=np.where(np.isin([str(i) for i in keyfileArray[:,0]],notPresentIDs))
    cleanedKeyfile=np.delete(keyfileArray,removeRows,axis=0)
    ungroupedLabelVec=np.repeat('ungrouped',len(ungroupedIDs))
    
    #switch to string, because why not
    cleanedKeyfile[:,0]=[str(i) for i in cleanedKeyfile[:,0]]
    
    ungroupedArray=np.column_stack([ungroupedIDs,ungroupedLabelVec])
    
    outArray=np.vstack([cleanedKeyfile,ungroupedArray])
    
    sortedGroupVec=[0]*len(outArray)
    
    for iSubjects in range(len(outArray)):
        subjectIndex=np.asarray(np.where(subjectsColArray==outArray[iSubjects,0]))[0]
        sortedGroupVec[subjectIndex[0]]=outArray[iSubjects,1]
        

    tractDataFrameOut=copy.deepcopy(tractDataFrameIn)
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
            currentDistance[0,iNodes]=wasserstein_distance(group1Data[:,iNodes],group2Data[:,iNodes])
            
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
    
        figPathName=saveFigDir+'/'+tractName+'_'+iComparisons+'.png'
    
        fig.savefig(figPathName,dpi=300)
        
        close() 
        
emdPlotsFromTractProfiles(projectDir,keyfilePath,tractName,metricLabel,iterNumber)    

projectDir='/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd'



#/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd/bids/GroupKey_v2.csv
keyfilePath='/Users/plab/Documents/tractProfiles/proj-5a74b26ab7f7fb00495482bd/bids/GroupKey_v2.csv'

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