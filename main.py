from HelpersX import *

path = r"C:\Users\Hossam\Downloads\Software\static\Features\Default_Dataset_All_Distances_No_Augmentation\Features.csv"
featuresFilePath = path
cols2Drop = [
  "Index", "ROI", "Mask", "NoPadROI",
  "NoPadMask", "Folder", "Date", "Timestamp",
  "IsAugmented", "Augmentation", "File",
]
# classifiersList = list(NAMES_CLASSIFIERS_DICT.keys())[:1]
classifiersList = ["DecisionTreeClassifier"]
scalersList = SCALERS[:2]

StageClassification(
  featuresFilePath,
  cols2Drop,
  classifiersList,
  scalersList,
  testRatio=0.2,
  outliersDetection=True,
  outliersFraction=0.15,
  featuresSelection=True,
  noOfTrials=5,
  correlationThreshold=0.90,
  randomState=42,
  stage=2,
  level="Down",
)

# firstStageOverallHistory = os.path.join(path, "First Stage", "Overall_History.csv")
# firstStageOverallPickle = os.path.join(path, "First Stage", "Overall_History.pickle")
#
# df = pd.read_csv(firstStageOverallHistory)
# with open(firstStageOverallPickle, "rb") as f:
#   data = pickle.load(f)
#
# print(df.columns)
#
# groupByClassifier = df.groupby("Classifier")
# # Loop on each group.
# for classifier, group in groupByClassifier:
#   print(classifier)
#   print(group)
