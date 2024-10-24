from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session, jsonify
import flask, flask_wtf
from flask_wtf.csrf import CSRFError, CSRFProtect
from HelpersX import *

# from Helpers import *

print("- Flask:", flask.__version__)
print("- Flask WTF:", flask_wtf.__version__)

IMAGES, MASKS, CLASSES, UNIQUE_CLASSES, FILES = [], [], [], [], []
CURRENT_TIMESTAMP = int(time.time())
CURRENT_DATE = f"{time.strftime('%Y-%m-%d', time.localtime(CURRENT_TIMESTAMP))}"
CURRENT_DATETIME = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(CURRENT_TIMESTAMP))}"
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]
FE_PROGRESS = 0  # Features Extraction Progress.
PP_PROGRESS = 0  # Preprocessing Progress.
CL_PROGRESS = 0  # Classification Progress.
IN_PROGRESS = 0  # Classification Progress.
RANDOM_STATE = 42  # Random State for reproducibility.
COLS_TO_DROP = [
  "Index", "ROI", "Mask", "NoPadROI",
  "NoPadMask", "Folder", "Date", "Timestamp",
  "IsAugmented", "Augmentation", "File",
]

LINE_STYLES = [
  "-", "--", "-.", ":",
  "-", "--", "-.", ":",
  "-", "--", "-.", ":",
]

# Seed all random number generators.
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/images"
app.config["PICKLES_FOLDER"] = "static/Pickles"
app.config["FEATURES_FOLDER"] = "static/Features"
app.config["CLASSIFICATION_FOLDER"] = "static/Classification"
app.config["LOGS_FOLDER"] = "static/Logs"
app.secret_key = "hmb&Super@secret!key"

csrf = CSRFProtect(app)

# Create a logger in a csv file with the name "App" and Timestamp.
logger = CreateLogger(app.config["LOGS_FOLDER"], "App", CURRENT_DATE)

# Log the start of the application and the timestamp.
logger.info("Application started.")
logger.info("Timestamp: " + str(CURRENT_TIMESTAMP))
logger.info("Date: " + CURRENT_DATE)
logger.info("Datetime: " + CURRENT_DATETIME)


# Log every request to the application.
@app.before_request
def logRequest():
  logger.info(f"Request: {request.url}")
  logger.info(f"Method: {request.method}")
  logger.info(f"Remote Address: {request.remote_addr}")
  logger.info(f"User Agent: {request.user_agent.string}")
  logger.info(f"Timestamp: {int(time.time())}")
  logger.info(f"Date: {time.strftime('%Y-%m-%d', time.localtime(int(time.time())))}")
  logger.info(f"Datetime: {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(time.time())))}")
  logger.info(f"Headers: {request.headers}")
  logger.info(f"Data: {request.data}")
  logger.info(f"Form: {request.form}")
  logger.info(f"Args: {request.args}")
  logger.info(f"Session: {session}")
  logger.info(f"Endpoint: {request.endpoint}")
  logger.info(f"Blueprint: {request.blueprint}")
  logger.info(f"View Args: {request.view_args}")
  logger.info(f"URL Rule: {request.url_rule}")
  logger.info(f"Max Content Length: {request.max_content_length}")
  logger.info(f"Content Length: {request.content_length}")
  logger.info(f"Content Type: {request.content_type}")
  logger.info(f"Content Encoding: {request.content_encoding}")
  logger.info(f"Content MD5: {request.content_md5}")


def GetStoredPickles():
  # Get all stored pickles.
  pickles = os.listdir(app.config["PICKLES_FOLDER"])
  pickles = [p for p in pickles if p.endswith(".pickle")]

  # Get the storage size of each pickle.
  pickleSizes = []
  for p in pickles:
    picklePath = os.path.join(app.config["PICKLES_FOLDER"], p)
    pickleSize = os.path.getsize(picklePath)
    pickleSizes.append(pickleSize)

  pickleSizes = [
    StorageSizeToString(size)
    for size in pickleSizes
  ]

  pickleZip = list(zip(pickles, pickleSizes))

  return pickleZip


@app.route("/")
def index():
  return render_template(
    "index.html",
  )


@app.route("/progress-process", methods=["GET"])
def ProgressProcess():
  global FE_PROGRESS, PP_PROGRESS, CL_PROGRESS, IN_PROGRESS

  flag = request.args.get("flag")
  progress = None
  if (flag == "features_extraction"):
    progress = FE_PROGRESS
  elif (flag == "preprocessing"):
    progress = PP_PROGRESS
  elif (flag == "classification"):
    progress = CL_PROGRESS
  elif (flag == "inference"):
    progress = IN_PROGRESS

  return jsonify(
    {
      "progress": progress,
      "message" : "Processing...",
      "status"  : "success",
    }
  )


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
  global IMAGES, MASKS, CLASSES, UNIQUE_CLASSES, PP_PROGRESS, FILES

  if (request.method == "POST"):
    PP_PROGRESS = 0

    # Load images and masks.
    if (request.form["action"] == "load"):
      try:
        baseFolder = request.form["baseFolder"]

        if (len(baseFolder) == 0):
          return jsonify(
            {
              "progress": PP_PROGRESS,
              "message" : "Please select a valid folder.",
              "status"  : "error",
            }
          )

        baseFolder = baseFolder.strip()
        imagesBasePath = os.path.join(baseFolder, "Original")
        masksBasePath = os.path.join(baseFolder, "Masks")

        if ((not os.path.exists(imagesBasePath)) or (not os.path.exists(masksBasePath))):
          return jsonify(
            {
              "progress": PP_PROGRESS,
              "message" : "Invalid folder structure. Make sure 'Original' and 'Masks' folders exist.",
              "status"  : "error",
            }
          )

        PP_PROGRESS = 5

        uniqueClasses = os.listdir(imagesBasePath)
        images, masks, classes, files = [], [], [], []

        totalNumberOfImages = 0
        for cls in uniqueClasses:
          imagesPath = os.path.join(imagesBasePath, cls)
          imagesFiles = os.listdir(imagesPath)
          totalNumberOfImages += len(imagesFiles)

        percentagePerImage = np.round(90.0 / totalNumberOfImages, 2)

        for cls in uniqueClasses:
          imagesPath = os.path.join(imagesBasePath, cls)
          masksPath = os.path.join(masksBasePath, cls)

          if ((not os.path.exists(imagesPath)) or (not os.path.exists(masksPath))):
            return jsonify(
              {
                "progress": PP_PROGRESS,
                "message" : "The folder structure is not correct.",
                "status"  : "error",
              }
            )

          imagesFiles = sorted(os.listdir(imagesPath))
          # masksFiles = os.listdir(masksPath)

          for i in tqdm.tqdm(range(len(imagesFiles))):
            image = cv2.imread(
              os.path.join(imagesPath, imagesFiles[i]),
              cv2.IMREAD_COLOR,
            )
            mask = cv2.imread(
              os.path.join(masksPath, imagesFiles[i]),
              cv2.IMREAD_GRAYSCALE,
            )
            images.append(image)
            masks.append(mask)
            classes.append(cls)
            files.append(imagesFiles[i])

            PP_PROGRESS += percentagePerImage
            PP_PROGRESS = float(np.round(PP_PROGRESS, 3))

        PP_PROGRESS = 95

        images, masks = PreprocessImagesMasks(images, masks)
        IMAGES, MASKS, UNIQUE_CLASSES, CLASSES, FILES = images, masks, uniqueClasses, classes, files

        # Select a random image and mask to display.
        index = random.randint(0, len(images) - 1)
        image = images[index]
        mask = masks[index]

        # Display the image and mask side by side.
        plt.figure(figsize=(3, 4))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.tight_layout()
        combinedImagePath = os.path.join(app.config["UPLOAD_FOLDER"], "preprocessing_random.jpg")
        plt.savefig(
          combinedImagePath,
          bbox_inches="tight",
          dpi=300,
        )
        plt.close()
        gc.collect()

        PP_PROGRESS = 100

        imageBase64 = Image2Base64(combinedImagePath).decode("utf-8")

        return jsonify(
          {
            "progress"  : PP_PROGRESS,
            "message"   : "Images and masks loaded successfully in the memory!",
            "status"    : "success",
            "image"     : imageBase64,
            "numImages" : len(images),
            "numMasks"  : len(masks),
            "numClasses": len(uniqueClasses),
            "baseFolder": baseFolder,
          }
        )
      except Exception as ex:
        logger.info(f"[Preprocessing Phase] Error: {str(ex)}")
        logger.info(f"[Preprocessing Phase] Error: {traceback.format_exc()}")
        return jsonify(
          {
            "progress": PP_PROGRESS,
            "message" : "An error occurred while loading images and masks.",
            "status"  : "error",
          }
        )
    # Save data to pickle file.
    elif (request.form["action"] == "save"):
      try:
        if (len(IMAGES) == 0 or len(MASKS) == 0):
          return jsonify(
            {
              "progress": PP_PROGRESS,
              "message" : "Please load images and masks first.",
              "status"  : "error",
            }
          )

        baseFolder = request.form["baseFolder"]
        folderName = os.path.basename(baseFolder)
        currentDT = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(time.time())))}"
        fileName = f"{folderName}_{len(IMAGES)}_{currentDT}.pickle"
        savePath = os.path.join(
          app.config["PICKLES_FOLDER"],
          fileName,
        )

        PP_PROGRESS = 50

        with open(savePath, "wb") as f:
          # X = (np.array(IMAGES) / 255.0).astype(np.float32)
          # Y = (np.array(MASKS) / 255.0).astype(np.float32)
          X = np.array(IMAGES)
          Y = np.array(MASKS)
          pickle.dump((X, Y, CLASSES, FILES, UNIQUE_CLASSES), f)

        pickleZip = GetStoredPickles()

        PP_PROGRESS = 100

        return jsonify(
          {
            "progress" : PP_PROGRESS,
            "message"  : f"Data saved successfully to {fileName}.",
            "status"   : "success",
            "pickleZip": pickleZip,
          }
        )
      except Exception as ex:
        logger.info(f"[Preprocessing Phase] Error: {str(ex)}")
        logger.info(f"[Preprocessing Phase] Error: {traceback.format_exc()}")
        return jsonify(
          {
            "progress": PP_PROGRESS,
            "message" : "An error occurred while saving the data.",
            "status"  : "error",
          }
        )

  elif (request.method == "GET"):
    pickleZip = GetStoredPickles()

    preprocessingImage = Image2Base64("static/images/Preprocessing.jpg").decode("utf-8")

    # Render with form data if error occurred.
    baseFolder = session.pop("baseFolder", "")
    imageURL = session.pop("imageURL", "")
    return render_template(
      "preprocessing.html",
      baseFolder=baseFolder,
      imageURL=imageURL,
      pickleZip=pickleZip,
      preprocessingImage=preprocessingImage,
    )


@app.route("/features-extraction", methods=["GET", "POST"])
def featuresExtraction():
  global IMAGES, MASKS, CLASSES, FILES, CURRENT_TIMESTAMP, FE_PROGRESS

  if (request.method == "POST"):
    if (FE_PROGRESS > 0 and FE_PROGRESS < 100):
      return jsonify(
        {
          "progress": FE_PROGRESS,
          "message" : "There is an ongoing task. Please wait...",
          "status"  : "error",
        }
      )
    else:
      FE_PROGRESS = 0

  if (request.method == "POST"):
    try:
      FE_PROGRESS = 1

      pickleFile = request.form["pickleFile"]
      distances = request.form["distances"]
      augmentationTechs = request.form["augmentation"]
      fileName = request.form["fileName"]

      filePath = os.path.join(app.config["FEATURES_FOLDER"], fileName)
      picklePath = os.path.join(app.config["PICKLES_FOLDER"], pickleFile)

      if (len(pickleFile) == 0):
        FE_PROGRESS = 0
        return jsonify(
          {
            "progress": FE_PROGRESS,
            "message" : "Please select a valid pickle file.",
            "status"  : "error",
          }
        )

      if (len(fileName) == 0):
        FE_PROGRESS = 0
        return jsonify(
          {
            "progress": FE_PROGRESS,
            "message" : "Please enter a valid file name.",
            "status"  : "error",
          }
        )

      if (os.path.exists(filePath)):
        FE_PROGRESS = 0
        return jsonify(
          {
            "progress": FE_PROGRESS,
            "message" : "A folder with the same name already exists. Select a different name.",
            "status"  : "error",
          }
        )

      if (len(distances) == 0):
        FE_PROGRESS = 0
        return jsonify(
          {
            "progress": FE_PROGRESS,
            "message" : "Please enter a valid distance value.",
            "status"  : "error",
          }
        )

      if (not os.path.exists(picklePath)):
        FE_PROGRESS = 0
        return jsonify(
          {
            "progress": FE_PROGRESS,
            "message" : "Invalid pickle file.",
            "status"  : "error",
          }
        )

      FE_PROGRESS = 5

      storagePath = os.path.join(
        app.config["FEATURES_FOLDER"],
        fileName,
      )
      os.makedirs(storagePath, exist_ok=True)

      distancesList = [int(d) for d in distances.split(",")]
      augmentationTechsList = augmentationTechs.split(",")
      augmentationTechsList = [el for el in augmentationTechsList if len(el) > 0]

      with open(picklePath, "rb") as f:
        images, masks, classes, files, uniqueClasses = pickle.load(f)

      # for distance in distancesList:
      #   for cls in uniqueClasses:
      #     os.makedirs(os.path.join(storagePath, f"Top_{distance}_{cls}"), exist_ok=True)
      #     os.makedirs(os.path.join(storagePath, f"Middle_{distance}_{cls}"), exist_ok=True)
      #     os.makedirs(os.path.join(storagePath, f"Bottom_{distance}_{cls}"), exist_ok=True)
      # os.makedirs(os.path.join(storagePath, f"Features"), exist_ok=True)

      FE_PROGRESS = 10

      percentagePerImage = np.round(88.0 / (float(len(images)) * len(distancesList) * 3), 2)

      allFeatures = []
      for i in tqdm.tqdm(range(len(images))):
        image = images[i].astype(np.uint8)
        mask = masks[i].astype(np.uint8)
        cls = classes[i]
        file = files[i]

        if ((image.shape[0] != mask.shape[0]) or (image.shape[1] != mask.shape[1])):
          # Log with the name of the function and the error message.
          logger.info(
            f"[Features Extraction Phase] Error: Image and mask shapes do not match at index {i}."
          )
          continue

        mask = Gray2Binary(mask)
        maskCopy = mask.copy()

        # Find the contours.
        cnts = cv2.findContours(maskCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea)

        if (len(cnts) <= 0):
          logger.info(
            f"[Features Extraction Phase] Error: No contours found at index {i}."
          )
          continue
        if (len(cnts) != 3):
          logger.info(
            f"[Features Extraction Phase] Error: Number of contours is not 3 at index {i}."
          )
          continue

        try:
          for j, c in enumerate(cnts):
            folder = "Bottom"
            if (j == 1):
              folder = "Middle"
            elif (j == 2):
              folder = "Top"

            featuresDict = CalculateContourFeatures(c)

            subMaskCopy = np.zeros_like(mask)
            cv2.fillPoly(subMaskCopy, [c], color=(255, 255, 255))
            distanceMap = CalculateDistanceMap(subMaskCopy)
            quantiles = np.quantile(
              sorted(set(distanceMap.flatten())),
              np.array(distancesList) / 100.0,
            )
            for q, quantile in enumerate(quantiles):
              subFeatures = {}

              temp = subMaskCopy.copy()
              temp[distanceMap > quantile] = 0
              distance = distancesList[q]
              roi = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), temp)

              data = [(roi, temp)]
              if (len(augmentationTechsList) > 0):
                rndChoice, roiAug, tempAug = RandomAugmentation(roi, temp, augmentationTechsList)
                data.append((roiAug, tempAug))
              else:
                rndChoice, roiAug, tempAug = None, None, None

              augExt = "NoAug"

              for z, (a, b) in enumerate(data):
                roiNoPad = RemovePadding(a).astype(np.uint8)
                tempNoPad = RemovePadding(b).astype(np.uint8)

                firstK, firstV = RadiomicsFeaturesHelper().CalculateFirstOrderFeatures(
                  roiNoPad,
                  isSingleObject=True,
                  returnSqueezedRowDict=True,
                  returnKeyValuesLists=True,
                  discardKeys=["EmpiricalCumulativeDistributionFunction"],
                )

                glrlm = RadiomicsFeaturesHelper().CalculateMultiGLRLM(
                  roiNoPad,
                  angles=[0, 45, 90, 135],  # 4 Angles.
                  includeZero=True,
                  normalize=True,
                  applyAverage=True,
                  grayLevels=(0, 255),
                )

                glcm = RadiomicsFeaturesHelper().CalculateMultiGLCM(
                  roiNoPad,
                  angles=[0, 45, 90, 135],  # 4 Angles.
                  distances=[1, 3, 5],  # 3 Distances.
                  normalize=True,
                  applyAverage=True,
                  grayLevels=(0, 255),
                )

                glrlmFeatures = RadiomicsFeaturesHelper().CalculateGLRLMAllFeatures(glrlm)
                glcmFeatures = RadiomicsFeaturesHelper().CalculateGLCMAllFeatures(glcm)

                for ii, kk in enumerate(firstK):
                  subFeatures[kk] = firstV[ii]
                for kk in glcmFeatures.keys():
                  subFeatures[kk] = glcmFeatures[kk]
                for kk in glrlmFeatures.keys():
                  subFeatures[kk] = glrlmFeatures[kk]
                for kk in featuresDict.keys():
                  subFeatures[kk] = featuresDict[kk]

                # cv2.imwrite(
                #   os.path.join(
                #     storagePath,
                #     f"{folder}_{distance}_{cls}", f"Mask_{augExt}_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                #   ),
                #   b,
                # )
                #
                # cv2.imwrite(
                #   os.path.join(
                #     storagePath,
                #     f"{folder}_{distance}_{cls}", f"ROI_{augExt}_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                #   ),
                #   a,
                # )
                #
                # cv2.imwrite(
                #   os.path.join(
                #     storagePath,
                #     f"{folder}_{distance}_{cls}", f"NoPadMask_{augExt}_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                #   ),
                #   tempNoPad,
                # )
                #
                # cv2.imwrite(
                #   os.path.join(
                #     storagePath,
                #     f"{folder}_{distance}_{cls}", f"NoPadROI_{augExt}_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                #   ),
                #   roiNoPad,
                # )

                selectedAugTech = None
                if (rndChoice is not None):
                  selectedAugTech = rndChoice

                subFeatures["Index"] = i
                subFeatures["Quantile"] = distance
                subFeatures["Position"] = folder
                subFeatures["Class"] = cls
                subFeatures["File"] = file
                subFeatures["IsAugmented"] = augExt
                subFeatures["Augmentation"] = selectedAugTech if (selectedAugTech is not None) else "NoAug"
                # subFeatures["ROI"] = f"ROI_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                # subFeatures["Mask"] = f"Mask_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                # subFeatures["NoPadROI"] = f"NoPadROI_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                # subFeatures["NoPadMask"] = f"NoPadMask_{cls}_I_{i}_P_{j}_D_{q}.jpg"
                # subFeatures["Folder"] = f"{folder}_{distance}_{cls}"
                subFeatures["Date"] = CURRENT_DATETIME
                subFeatures["Timestamp"] = CURRENT_TIMESTAMP

                allFeatures.append(subFeatures)

                augExt = "Aug"

              FE_PROGRESS += percentagePerImage
              FE_PROGRESS = float(np.round(FE_PROGRESS, 3))
        except Exception as ex:
          print(ex)
          logger.info(
            f"[Features Extraction Phase] Error: {str(ex)} at index {i}."
          )
          # Log the ex line.
          logger.info(
            f"[Features Extraction Phase] Error: {traceback.format_exc()}"
          )
          continue

      df = pd.DataFrame(allFeatures)
      df.to_csv(
        os.path.join(storagePath, "Features.csv"),
        index=False,
      )

      configurations = {
        "FileName"                         : fileName,
        "Pickle File"                      : pickleFile,
        "Distances"                        : distances,
        "Augmentation"                     : augmentationTechs,
        "Timestamp"                        : CURRENT_TIMESTAMP,
        "Datetime"                         : CURRENT_DATETIME,
        "Date"                             : CURRENT_DATE,
        "Features"                         : df.columns,
        "Unique Classes"                   : uniqueClasses,
        "Number of Images"                 : len(images),
        "Number of Masks"                  : len(masks),
        "Number of Files"                  : len(files),
        "Number of Features"               : len(allFeatures),
        "Number of Distances"              : len(distancesList),
        "Number of Augmentation Techniques": len(augmentationTechsList),
        "Number of Classes"                : len(CLASSES),
        "Number of Unique Classes"         : len(uniqueClasses),
        "Number of Features Columns"       : len(df.columns),
      }

      with open(os.path.join(storagePath, "Configurations.pickle"), "wb") as f:
        pickle.dump(configurations, f)

      FE_PROGRESS = 100

      return jsonify(
        {
          "progress": FE_PROGRESS,
          "message" : "Task completed successfully.",
          "status"  : "success",
        }
      )
    except Exception as ex:
      logger.info(f"[Features Extraction Phase] Error: {str(ex)}")
      logger.info(f"[Features Extraction Phase] Error: {traceback.format_exc()}")
      return jsonify(
        {
          "progress": FE_PROGRESS,
          "message" : "An error occurred while extracting features.",
          "status"  : "error",
        }
      )

  elif (request.method == "GET"):
    pickleZip = GetStoredPickles()

    markersImage = Image2Base64("static/images/Markers.jpg").decode("utf-8")
    markersExtractionImage = Image2Base64("static/images/Markers Extraction.jpg").decode("utf-8")

    return render_template(
      "features_extraction.html",
      pickleZip=pickleZip,
      markersImage=markersImage,
      markersExtractionImage=markersExtractionImage,
    )


@app.route("/features-navigation", methods=["GET"])
def featuresNavigation():
  featuresPath = app.config["FEATURES_FOLDER"]
  featuresFolders = os.listdir(featuresPath)
  featuresFolders = [f for f in featuresFolders if os.path.isdir(os.path.join(featuresPath, f))]
  featuresFolders = sorted(featuresFolders)

  markersImage = Image2Base64("static/images/Markers.jpg").decode("utf-8")
  markersExtractionImage = Image2Base64("static/images/Markers Extraction.jpg").decode("utf-8")

  featuresContent = []
  for folder in featuresFolders:
    folderPath = os.path.join(featuresPath, folder, "Features.csv")
    if (not os.path.exists(folderPath)):
      continue
    folderSplit = folder.split("_")
    pickleFileName = f"{folderSplit[0]}_{folderSplit[1]}_{folderSplit[2]}.pickle"
    # distances = [
    #   el for el in os.listdir(os.path.join(featuresPath, folder))
    #   if (el.startswith("Top") or el.startswith("Middle") or el.startswith("Bottom"))
    # ]
    # distances = list(set([el.split("_")[1] for el in distances]))
    # distancesList = [int(d) for d in distances]
    configurationsFile = os.path.join(featuresPath, folder, "Configurations.pickle")
    if (not os.path.exists(configurationsFile)):
      continue
    with open(configurationsFile, "rb") as f:
      configurations = pickle.load(f)
    numOfRecords = configurations["Number of Features"]
    fileCreationDate = os.path.getctime(folderPath)
    fileCreationDateFormatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fileCreationDate))
    data = pd.read_csv(folderPath)
    featuresColumns = data.columns.tolist()[:-7]
    top10Records = data[featuresColumns].head(10)
    top10RecordsList = top10Records.values.tolist()
    distancesList = configurations["Distances"]

    if (os.path.exists(folderPath)):
      featuresContent.append({
        "folder"                        : folder,
        "pickleFileName"                : pickleFileName,
        "distances"                     : [
          el for el in distancesList.split(",")
          if len(el) > 0
        ],
        "numOfRecords"                  : numOfRecords,
        "top10Records"                  : top10RecordsList,
        "fileCreationDate"              : fileCreationDateFormatted,
        "featuresColumns"               : featuresColumns,
        "numberOfClasses"               : configurations["Number of Unique Classes"],
        "numberOfImages"                : configurations["Number of Images"],
        "numberOfMasks"                 : configurations["Number of Masks"],
        "numberOfFiles"                 : configurations["Number of Files"],
        "numberOfFeatures"              : configurations["Number of Features"],
        "numberOfDistances"             : configurations["Number of Distances"],
        "numberOfAugmentationTechniques": configurations["Number of Augmentation Techniques"],
        "augmentationTechniques"        : [
          el for el in configurations["Augmentation"].split(",")
          if len(el) > 0
        ],
        "classes"                       : configurations["Unique Classes"]
      })

  return render_template(
    "features_navigation.html",
    featuresContent=featuresContent,
    markersImage=markersImage,
    markersExtractionImage=markersExtractionImage,
  )


@app.route("/classification", methods=["GET", "POST"])
def classification():
  global CL_PROGRESS, RANDOM_STATE, COLS_TO_DROP

  if (request.method == "POST"):
    if (CL_PROGRESS > 0 and CL_PROGRESS < 100):
      return jsonify(
        {
          "progress": CL_PROGRESS,
          "message" : "There is an ongoing task. Please wait...",
          "status"  : "error",
        }
      )
    else:
      CL_PROGRESS = 0

  if (request.method == "POST"):
    try:
      featuresFile = request.form["featuresFile"]
      trainToTestRatio = request.form["trainTestRatio"]
      outliersDetection = request.form["outliersDetection"]
      outliersFraction = request.form["outliersFraction"]
      featuresSelection = request.form["featuresSelection"]
      correlationThreshold = request.form["correlationThreshold"]
      scaling = request.form["scaling"]
      classifiers = request.form["classifiers"]
      noOfTrials = request.form["noOfTrials"]

      if (len(featuresFile) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please select a valid features file.",
            "status"  : "error",
          }
        )

      if (len(trainToTestRatio) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please enter a valid train to test ratio.",
            "status"  : "error",
          }
        )

      if (len(outliersDetection) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please select whether to use outliers detection or not.",
            "status"  : "error",
          }
        )

      if (len(outliersFraction) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please enter a valid outliers fraction.",
            "status"  : "error",
          }
        )

      if (len(featuresSelection) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please select whether to use features selection or not.",
            "status"  : "error",
          }
        )

      if (len(correlationThreshold) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please enter a valid correlation threshold.",
            "status"  : "error",
          }
        )

      if (len(scaling) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please select the scaling techniques.",
            "status"  : "error",
          }
        )

      if (len(classifiers) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please select the classifiers.",
            "status"  : "error",
          }
        )

      if (len(noOfTrials) == 0):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Please enter a valid number of trials.",
            "status"  : "error",
          }
        )

      featuresFilePath = os.path.join(app.config["FEATURES_FOLDER"], featuresFile, "Features.csv")
      configurationsFilePath = os.path.join(app.config["FEATURES_FOLDER"], featuresFile, "Configurations.pickle")

      if (not os.path.exists(featuresFilePath) or not os.path.exists(configurationsFilePath)):
        CL_PROGRESS = 0
        return jsonify(
          {
            "progress": CL_PROGRESS,
            "message" : "Invalid features file.",
            "status"  : "error",
          }
        )

      CL_PROGRESS = 3  # After validation.

      currentDT = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(time.time())))}"
      testRatio = float(1.0 - (float(trainToTestRatio) / 100.0))
      classifiersList = [el for el in classifiers.split(",") if (len(el) > 0)]
      scalersList = [el for el in scaling.split(",") if (len(el) > 0)]

      clsFolder = os.path.join(
        app.config["CLASSIFICATION_FOLDER"],
        featuresFile,
        f"History-{currentDT}",
      )

      cmFolder = os.path.join(
        app.config["CLASSIFICATION_FOLDER"],
        featuresFile,
        f"History-{currentDT}",
        "Confusion Matrices",
      )
      rocFolder = os.path.join(
        app.config["CLASSIFICATION_FOLDER"],
        featuresFile,
        f"History-{currentDT}",
        "ROC AUC Curves",
      )
      metaFolder = os.path.join(
        app.config["CLASSIFICATION_FOLDER"],
        featuresFile,
        f"History-{currentDT}",
        "Meta Results",
      )
      latexFolder = os.path.join(
        app.config["CLASSIFICATION_FOLDER"],
        featuresFile,
        f"History-{currentDT}",
        "LaTeX Results",
      )

      os.makedirs(clsFolder, exist_ok=True)
      os.makedirs(cmFolder, exist_ok=True)
      os.makedirs(rocFolder, exist_ok=True)
      os.makedirs(metaFolder, exist_ok=True)
      os.makedirs(latexFolder, exist_ok=True)

      CL_PROGRESS = 5  # After Configurations.

      history, dfStores, allPredictionsDF, allMetricsDF, allCMs = StageClassification(
        featuresFilePath,
        COLS_TO_DROP,
        classifiersList,
        scalersList,
        testRatio=testRatio,
        outliersDetection=(outliersDetection == "1"),
        outliersFraction=float(int(outliersFraction) / 100.0),
        featuresSelection=(featuresSelection == "1"),
        noOfTrials=int(noOfTrials),
        correlationThreshold=float(int(correlationThreshold) / 100.0),
        randomState=RANDOM_STATE,
        stage=1,
        level="",
      )

      # Display the CMs as plots.
      for cmKey in allCMs.keys():
        cm = allCMs[cmKey]
        cmPlt = PlotConfusionMatrix(
          cm,
          history[list(history.keys())[0]]["Unique Classes"],
          normalize=False,
          title=None,
          cmap="flare"
        )
        cmPlt.savefig(
          os.path.join(cmFolder, f"First_Stage_{cmKey}_CM.pdf"),
          bbox_inches="tight",
          dpi=300,
        )
        plt.close()
        gc.collect()

      with open(os.path.join(metaFolder, "First_Stage_History.pickle"), "wb") as f:
        pickle.dump((history, dfStores, allPredictionsDF, allMetricsDF, allCMs), f)
      for dfStore in dfStores:
        dfStore[1].to_csv(os.path.join(metaFolder, dfStore[0]), index=False)
      allPredictionsDF.to_csv(os.path.join(metaFolder, "First_Stage_All_Predictions.csv"), index=False)
      allMetricsDF.to_csv(os.path.join(metaFolder, "First_Stage_All_Metrics.csv"), index=False)

      CL_PROGRESS = 40  # After Classification (First Stage).

      history, dfStores, allPredictionsDF, allMetricsDF, allCMs = StageClassification(
        featuresFilePath,
        COLS_TO_DROP,
        classifiersList,
        scalersList,
        testRatio=testRatio,
        outliersDetection=(outliersDetection == "1"),
        outliersFraction=float(int(outliersFraction) / 100.0),
        featuresSelection=(featuresSelection == "1"),
        noOfTrials=int(noOfTrials),
        correlationThreshold=float(int(correlationThreshold) / 100.0),
        randomState=RANDOM_STATE,
        stage=2,
        level="Up",
      )

      # Display the CMs as plots.
      for cmKey in allCMs.keys():
        cm = allCMs[cmKey]
        cmPlt = PlotConfusionMatrix(
          cm,
          history[list(history.keys())[0]]["Unique Classes"],
          normalize=False,
          title=None,
          cmap="flare"
        )
        cmPlt.savefig(
          os.path.join(cmFolder, f"Second_Stage_Up_{cmKey}_CM.pdf"),
          bbox_inches="tight",
          dpi=300,
        )
        plt.close()
        gc.collect()

      with open(os.path.join(metaFolder, "Second_Stage_Up_History.pickle"), "wb") as f:
        pickle.dump((history, dfStores, allPredictionsDF, allMetricsDF, allCMs), f)
      for dfStore in dfStores:
        dfStore[1].to_csv(os.path.join(metaFolder, dfStore[0]), index=False)
      allPredictionsDF.to_csv(os.path.join(metaFolder, "Second_Stage_Up_All_Predictions.csv"), index=False)
      allMetricsDF.to_csv(os.path.join(metaFolder, "Second_Stage_Up_All_Metrics.csv"), index=False)

      CL_PROGRESS = 75  # After Classification (Second Stage - Up Level).

      history, dfStores, allPredictionsDF, allMetricsDF, allCMs = StageClassification(
        featuresFilePath,
        COLS_TO_DROP,
        classifiersList,
        scalersList,
        testRatio=testRatio,
        outliersDetection=(outliersDetection == "1"),
        outliersFraction=float(int(outliersFraction) / 100.0),
        featuresSelection=(featuresSelection == "1"),
        noOfTrials=int(noOfTrials),
        correlationThreshold=float(int(correlationThreshold) / 100.0),
        randomState=RANDOM_STATE,
        stage=2,
        level="Down",
      )

      # Display the CMs as plots.
      for cmKey in allCMs.keys():
        cm = allCMs[cmKey]
        cmPlt = PlotConfusionMatrix(
          cm,
          history[list(history.keys())[0]]["Unique Classes"],
          normalize=False,
          title=None,
          cmap="flare"
        )
        cmPlt.savefig(
          os.path.join(cmFolder, f"Second_Stage_Down_{cmKey}_CM.pdf"),
          bbox_inches="tight",
          dpi=300,
        )
        plt.close()
        gc.collect()

      with open(os.path.join(metaFolder, "Second_Stage_Down_History.pickle"), "wb") as f:
        pickle.dump((history, dfStores, allPredictionsDF, allMetricsDF, allCMs), f)
      for dfStore in dfStores:
        dfStore[1].to_csv(os.path.join(metaFolder, dfStore[0]), index=False)
      allPredictionsDF.to_csv(os.path.join(metaFolder, "Second_Stage_Down_All_Predictions.csv"), index=False)
      allMetricsDF.to_csv(os.path.join(metaFolder, "Second_Stage_Down_All_Metrics.csv"), index=False)

      CL_PROGRESS = 90  # After Classification (Second Stage - Down Level).

      # Post-processing after classification.

      firstStagePredictions = pd.read_csv(os.path.join(metaFolder, "First_Stage_All_Predictions.csv"))
      secondStageUpPredictions = pd.read_csv(os.path.join(metaFolder, "Second_Stage_Up_All_Predictions.csv"))
      secondStageDownPredictions = pd.read_csv(os.path.join(metaFolder, "Second_Stage_Down_All_Predictions.csv"))

      bothStagesPredictions = firstStagePredictions.copy()
      for i in range(len(bothStagesPredictions)):
        if (bothStagesPredictions["File"][i] in secondStageUpPredictions["File"].values):
          idx = secondStageUpPredictions.index[
            secondStageUpPredictions["File"] == bothStagesPredictions["File"][i]
            ].tolist()[0]
          row = secondStageUpPredictions.iloc[idx]
          for el in row.index:
            bothStagesPredictions.at[i, f"Second_Stage_{el}"] = row[el]
        if (bothStagesPredictions["File"][i] in secondStageDownPredictions["File"].values):
          idx = secondStageDownPredictions.index[
            secondStageDownPredictions["File"] == bothStagesPredictions["File"][i]
            ].tolist()[0]
          row = secondStageDownPredictions.iloc[idx]
          for el in row.index:
            bothStagesPredictions.at[i, f"Second_Stage_{el}"] = row[el]

      bothStagesPredictions.to_csv(os.path.join(metaFolder, "Both_Stages_All_Predictions.csv"), index=False)

      # Calculate metrics for both stages after applying no error propagation.
      # No EP means that the second stage predictions are applied only on the samples that were
      # correctly predicted in the first stage.
      metricsBothStagesNoErrorProp = []
      for col in firstStagePredictions.columns:
        if (col.endswith("_Majority")):
          actual = firstStagePredictions["Class"]
          predicted = firstStagePredictions[col]
          areSimilar = (actual == predicted)
          actualOriginal = firstStagePredictions["Class_Original"]
          actualOriginal = actualOriginal[areSimilar]
          pred2nd = bothStagesPredictions[f"Second_Stage_{col}"]
          pred2nd = pred2nd[areSimilar]
          uniqueClassesInActual = list(actualOriginal.unique())
          uniqueClassesInPred = list(pred2nd.unique())
          if (len(uniqueClassesInActual) != len(uniqueClassesInPred)):
            print(f"Classes are not the same for {col}.")
            continue
          metrics, cm, fpr, tpr, aucValuesDict = CalculateMetrics(actualOriginal, pred2nd)

          cmPlt = PlotConfusionMatrix(
            cm,
            list(actualOriginal.unique()),
            normalize=False,
            title=None,
            cmap="Blues"
          )
          cmPlt.savefig(
            os.path.join(cmFolder, f"Both_Stages_No_EP_{col}_CM.pdf"),
            bbox_inches="tight",
            dpi=300,
          )
          plt.close()
          gc.collect()

          # fpr, tpr, aucValue = CalculateAUCROC(actualOriginal, pred2nd)
          # metrics["AUC"] = aucValue
          metrics["Column"] = col
          metricsBothStagesNoErrorProp.append(metrics)
          # Plot the ROC curve.
          plt.figure(figsize=(7, 7))
          for i, k in enumerate(aucValuesDict.keys()):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{k} AUC: {aucValuesDict[k]:.3f}", linestyle=LINE_STYLES[i])
          plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.0])
          plt.grid(True)
          plt.xlabel("False Positive Rate", fontsize=14)
          plt.ylabel("True Positive Rate", fontsize=14)
          # plt.title(f"ROC Curve ({col})", fontsize=14)
          plt.legend(loc="lower right", fontsize=14)
          plt.tight_layout()
          plt.savefig(
            os.path.join(rocFolder, f"Both_Stages_No_EP_{col}_ROC.pdf"),
            bbox_inches="tight",
            dpi=300,
          )
          plt.close()
          gc.collect()

      # Calculate metrics for both stages after applying error propagation.
      # EP means that the second stage predictions are applied on all samples.
      metricsBothStagesErrorProp = []
      for col in firstStagePredictions.columns:
        if (col.endswith("_Majority")):
          actual = firstStagePredictions["Class"]
          predicted = firstStagePredictions[col]
          actualOriginal = firstStagePredictions["Class_Original"]
          pred2nd = bothStagesPredictions[f"Second_Stage_{col}"]
          uniqueClassesInActual = list(actualOriginal.unique())
          uniqueClassesInPred = list(pred2nd.unique())
          if (len(uniqueClassesInActual) != len(uniqueClassesInPred)):
            print(f"Classes are not the same for {col}.")
            continue
          metrics, cm, fpr, tpr, aucValuesDict = CalculateMetrics(actualOriginal, pred2nd)

          cmPlt = PlotConfusionMatrix(
            cm,
            list(actualOriginal.unique()),
            normalize=False,
            title=None,
            cmap="Blues"
          )
          cmPlt.savefig(
            os.path.join(cmFolder, f"Both_Stages_EP_{col}_CM.pdf"),
            bbox_inches="tight",
            dpi=300,
          )
          plt.close()
          gc.collect()

          # fpr, tpr, aucValue = CalculateAUCROC(actualOriginal, pred2nd)
          # metrics["AUC"] = aucValue
          metrics["Column"] = col
          metricsBothStagesErrorProp.append(metrics)
          # Plot the ROC curve.
          plt.figure(figsize=(7, 7))
          for i, k in enumerate(aucValuesDict.keys()):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{k} AUC: {aucValuesDict[k]:.3f}", linestyle=LINE_STYLES[i])
          plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.0])
          plt.grid(True)
          plt.xlabel("False Positive Rate", fontsize=14)
          plt.ylabel("True Positive Rate", fontsize=14)
          # plt.title(f"ROC Curve ({col})", fontsize=14)
          plt.legend(loc="lower right", fontsize=14)
          plt.tight_layout()
          plt.savefig(
            os.path.join(rocFolder, f"Both_Stages_EP_{col}_ROC.pdf"),
            bbox_inches="tight",
            dpi=300,
          )
          plt.close()
          gc.collect()

      with open(os.path.join(metaFolder, "Both_Stages_No_EP_Metrics.pickle"), "wb") as f:
        pickle.dump(metricsBothStagesNoErrorProp, f)

      with open(os.path.join(metaFolder, "Both_Stages_EP_Metrics.pickle"), "wb") as f:
        pickle.dump(metricsBothStagesErrorProp, f)

      metricsNoEPDF = pd.DataFrame(metricsBothStagesNoErrorProp)
      metricsNoEPDF.to_csv(os.path.join(metaFolder, "Both_Stages_No_EP_Metrics.csv"), index=False)

      metricsEPDF = pd.DataFrame(metricsBothStagesErrorProp)
      metricsEPDF.to_csv(os.path.join(metaFolder, "Both_Stages_EP_Metrics.csv"), index=False)

      CL_PROGRESS = 93

      # ============================================================================================= #

      columns2Print = [
        "Classifier",
        "Scaler",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "IoU",
        "BAC",
        "MCC",
        "Youden",
        "Yule",
        "AUC_Mean"
      ]

      allFinalists = {}

      for fileKeyword in ["First_Stage", "Second_Stage_Up", "Second_Stage_Down"]:

        stageMetrics = pd.read_csv(os.path.join(metaFolder, f"{fileKeyword}_All_Metrics.csv"))
        stagePredictions = pd.read_csv(os.path.join(metaFolder, f"{fileKeyword}_All_Predictions.csv"))

        allFinalists[f"{fileKeyword}_File"] = stagePredictions["File"].values
        allFinalists[f"{fileKeyword}_Class"] = stagePredictions["Class"].values
        allFinalists[f"{fileKeyword}_Class_Original"] = stagePredictions["Class_Original"].values

        # Delete rows that does not contain a keyword in the column.
        stageMetrics = stageMetrics[stageMetrics["Column"].str.contains("Majority")]
        # Get the classifier and scaler names.
        stageMetrics["Classifier"] = stageMetrics["Column"].apply(lambda x: x.split("_")[0])
        stageMetrics["Scaler"] = stageMetrics["Column"].apply(lambda x: x.split("_")[1])
        # Get the index of top-1 in each group.
        indexOfTop1 = stageMetrics.groupby("Classifier")["Accuracy"].idxmax()
        keywordsOfTop1 = stageMetrics.loc[indexOfTop1, "Column"].values

        # Add the predictions to the dictionary.
        # for keyword in keywordsOfTop1:
        #   allFinalists[keyword] = stagePredictions[keyword].values

        # Add Mean column.
        stageMetrics["Mean_Score"] = stageMetrics[columns2Print[2:]].mean(axis=1)
        # Get the top-1 metrics for each classifier.
        columns2PrintIndividual = columns2Print + ["Mean_Score"]
        top1Metrics = stageMetrics.loc[indexOfTop1, columns2PrintIndividual]
        # Sort the top-1 metrics by the mean score.
        top1Metrics.sort_values("Mean_Score", ascending=False, inplace=True)
        firstRowIndex = top1Metrics.index[0]
        # print(top1Metrics.loc[firstRowIndex])
        # Reset the index.
        top1Metrics.reset_index(drop=True, inplace=True)
        # Put the numbers as percentages with 2 decimal points.
        top1Metrics[columns2PrintIndividual[2:]] = top1Metrics[columns2PrintIndividual[2:]] * 100
        top1Metrics[columns2PrintIndividual[2:]] = top1Metrics[columns2PrintIndividual[2:]].applymap(
          lambda x: f"{x:.2f}")
        # Add "Order" column.
        top1Metrics["Order"] = np.arange(1, top1Metrics.shape[0] + 1)
        # print(top1Metrics)

        columnName = stageMetrics.loc[firstRowIndex, "Column"]
        allFinalists[f"{fileKeyword}_{columnName}"] = stagePredictions[columnName].values

        # Save the top-1 metrics to a CSV file.
        top1Metrics.to_csv(os.path.join(clsFolder, f"{fileKeyword}_Top1_Individual_Metrics.csv"), index=False)

        # majorityColumnsOnly = [c for c in stagePredictions.columns if c.endswith("Majority")]
        majorityColumnsOnly = keywordsOfTop1
        # print(majorityColumnsOnly)
        allCombinations = []
        for i in range(2, len(majorityColumnsOnly) + 1):
          allCombinations.extend(itertools.combinations(majorityColumnsOnly, i))
        # print(len(allCombinations))
        # print(allCombinations)

        combinationsHistory = []
        combinedPredictions = {}
        # Loop through all combinations.
        for i in tqdm.tqdm(range(len(allCombinations))):
          actual = stagePredictions["Class"]
          # Get the combination.
          combination = allCombinations[i]
          # Get the predictions.
          predictions = stagePredictions[list(combination)]
          # Get the majority votes using collections.Counter.
          majorityVotes = predictions.apply(lambda x: collections.Counter(x), axis=1)
          # Get the majority votes.
          majorityVotes = majorityVotes.apply(lambda x: x.most_common(1)[0][0])
          # Calculate the metrics.
          metrics, cm, fpr, tpr, aucValues = CalculateMetrics(actual, majorityVotes)
          metrics["Mean_Score"] = np.array([metrics[c] for c in columns2Print[2:-1]]).mean()
          metrics["Column"] = ", ".join([c for c in combination])
          metrics["Classifier"] = ", ".join([c.split("_")[0] for c in combination])
          metrics["Number of Classifiers"] = len(combination)
          combinationsHistory.append(metrics)
          combinedPredictions[metrics["Column"]] = majorityVotes.values

        combinationsHistoryDF = pd.DataFrame(combinationsHistory)
        combinationsHistoryDF.to_csv(
          os.path.join(metaFolder, f"{fileKeyword}_All_Top1_Combinations_Metrics.csv"), index=False)
        combinedPredictionsDF = pd.DataFrame(combinedPredictions)
        combinedPredictionsDF.to_csv(
          os.path.join(metaFolder, f"{fileKeyword}_All_Top1_Combinations_Predictions.csv"), index=False)

        # Get the index of top-1 in each group according to the number of classifiers.
        indexOfTop1 = combinationsHistoryDF.groupby("Number of Classifiers")["Accuracy"].idxmax()
        keywordsOfTop1 = combinationsHistoryDF.loc[indexOfTop1, "Column"].values

        # Add the predictions to the dictionary.
        # for keyword in keywordsOfTop1:
        #   allFinalists[keyword] = stagePredictions[keyword].values

        # Get the top-1 metrics for each classifier.
        columns2PrintComb = columns2Print[:1] + columns2Print[2:] + ["Mean_Score"] + ["Number of Classifiers"]
        top1MetricsComb = combinationsHistoryDF.loc[indexOfTop1, columns2PrintComb]
        # Sort the top-1 metrics by the mean score.
        top1MetricsComb.sort_values("Mean_Score", ascending=False, inplace=True)
        firstRowIndex = top1MetricsComb.index[0]
        # print(firstRowIndex)
        # Reset the index.
        top1MetricsComb.reset_index(drop=True, inplace=True)
        # Put the numbers as percentages with 2 decimal points.
        top1MetricsComb[columns2Print[2:] + ["Mean_Score"]] = top1MetricsComb[columns2Print[2:] + ["Mean_Score"]] * 100
        top1MetricsComb[columns2Print[2:] + ["Mean_Score"]] = top1MetricsComb[
          columns2Print[2:] + ["Mean_Score"]].applymap(
          lambda x: f"{x:.2f}"
        )
        # Add "Order" column.
        top1MetricsComb["Order"] = top1MetricsComb["Number of Classifiers"].apply(lambda x: f"Top-{x}")
        # Sort the top-1 metrics by the mean score.
        top1MetricsComb.sort_values("Number of Classifiers", ascending=True, inplace=True)
        top1MetricsComb.drop("Number of Classifiers", axis=1, inplace=True)
        # print(top1MetricsComb)

        columnName = combinationsHistoryDF.loc[firstRowIndex, "Column"]
        allFinalists[f"{fileKeyword}_{columnName}"] = combinedPredictions[columnName]

        # Save the top-1 metrics to a CSV file.
        top1MetricsComb.to_csv(os.path.join(clsFolder, f"{fileKeyword}_Top1_Top1_Combinations_Metrics.csv"),
                               index=False)

        # Print LaTeX table.
        latexContent = top1Metrics.style.hide(axis="index").to_latex(
          position_float="centering",
          caption=r"Stage Performance Summary: Individual Classifier Metrics.",
          multirow_align="c",
          multicol_align="c",
          column_format="|c" * (len(columns2Print) - 1) + "|",
          position="!htb",
          hrules=1,
          siunitx=True,

        )
        # print(latexContent)

        # Print LaTeX table.
        latexContentComb = top1MetricsComb.style.hide(axis="index").to_latex(
          position_float="centering",
          caption=r"Stage Performance Summary: Combined Classifier Metrics.",
          multirow_align="c",
          multicol_align="c",
          column_format="|c" * (len(columns2Print) - 1) + "|",
          position="!htb",
          hrules=1,
          siunitx=True,
        )
        # print(latexContentComb)

        # Save LaTeX table.
        with open(os.path.join(latexFolder, f"{fileKeyword}_Top1_Individual_Metrics.tex"), "w") as f:
          f.write(latexContent)
        with open(os.path.join(latexFolder, f"{fileKeyword}_Top1_Top1_Combinations_Metrics.tex"), "w") as f:
          f.write(latexContentComb)

        CL_PROGRESS += 1.5

      maxNumberOfRecords = max([len(allFinalists[c]) for c in allFinalists.keys()])
      for c in allFinalists.keys():
        if ("Second_Stage_Up" in c):
          allFinalists[c] = list(allFinalists[c]) + [np.nan] * (maxNumberOfRecords - len(allFinalists[c]))
        if ("Second_Stage_Down" in c):
          allFinalists[c] = [np.nan] * (maxNumberOfRecords - len(allFinalists[c])) + list(allFinalists[c])

      allFinalistsDF = pd.DataFrame(allFinalists)
      allFinalistsDF.to_csv(os.path.join(clsFolder, f"All_Stages_Top1_Predictions.csv"), index=False)

      firstStageSingleKeyword = [
        c for c in allFinalists.keys()
        if ("First_Stage" in c) and ("," not in c) and ("Majority" in c)
      ][0]
      secondStageUpSingleKeyword = [
        c for c in allFinalists.keys()
        if ("Second_Stage_Up" in c) and ("," not in c) and ("Majority" in c)
      ][0]
      secondStageDownSingleKeyword = [
        c for c in allFinalists.keys()
        if ("Second_Stage_Down" in c) and ("," not in c) and ("Majority" in c)
      ][0]
      firstStageCombinedKeyword = [
        c for c in allFinalists.keys()
        if ("First_Stage" in c) and ("," in c) and ("Majority" in c)
      ][0]
      secondStageUpCombinedKeyword = [
        c for c in allFinalists.keys()
        if ("Second_Stage_Up" in c) and ("," in c) and ("Majority" in c)
      ][0]
      secondStageDownCombinedKeyword = [
        c for c in allFinalists.keys()
        if ("Second_Stage_Down" in c) and ("," in c) and ("Majority" in c)
      ][0]
      classKeyword = [
        c for c in allFinalists.keys()
        if ("Class" in c) and ("Original" not in c)
      ][0]
      classOriginalKeyword = [
        c for c in allFinalists.keys()
        if ("Class" in c) and ("Original" in c)
      ][0]

      cls = allFinalists[classKeyword]
      clsOriginal = allFinalists[classOriginalKeyword]

      firstStageSingle = allFinalists[firstStageSingleKeyword]
      secondStageUpSingle = allFinalists[secondStageUpSingleKeyword]
      secondStageUpSingle = [c for c in secondStageUpSingle if str(c) != 'nan']
      secondStageDownSingle = allFinalists[secondStageDownSingleKeyword]
      secondStageDownSingle = [c for c in secondStageDownSingle if str(c) != 'nan']
      secondStageSingle = secondStageUpSingle + secondStageDownSingle

      metricsEP, cmEP, fprEP, tprEP, aucValuesEP = CalculateMetrics(clsOriginal, secondStageSingle)
      metricsEP["Mean_Score"] = np.array([metricsEP[c] for c in columns2Print[2:]]).mean()
      metricsEP["First_Stage_Keyword"] = firstStageSingleKeyword
      metricsEP["Second_Stage_Up_Keyword"] = secondStageUpSingleKeyword
      metricsEP["Second_Stage_Down_Keyword"] = secondStageDownSingleKeyword
      metricsEP["With Error Propagation"] = "Yes"

      secondStageNoEP = []
      clsOriginalNoEP = []
      for i in range(len(cls)):
        if (firstStageSingle[i] == cls[i]):
          secondStageNoEP.append(secondStageSingle[i])
          clsOriginalNoEP.append(clsOriginal[i])

      metricsNoEP, cmNoEP, fprNoEP, tprNoEP, aucValuesNoEP = CalculateMetrics(clsOriginalNoEP, secondStageNoEP)
      metricsNoEP["Mean_Score"] = np.array([metricsNoEP[c] for c in columns2Print[2:]]).mean()
      metricsNoEP["First_Stage_Keyword"] = firstStageSingleKeyword
      metricsNoEP["Second_Stage_Up_Keyword"] = secondStageUpSingleKeyword
      metricsNoEP["Second_Stage_Down_Keyword"] = secondStageDownSingleKeyword
      metricsNoEP["With Error Propagation"] = "No"

      firstStageCombined = allFinalists[firstStageCombinedKeyword]
      secondStageUpCombined = allFinalists[secondStageUpCombinedKeyword]
      secondStageUpCombined = [c for c in secondStageUpCombined if str(c) != 'nan']
      secondStageDownCombined = allFinalists[secondStageDownCombinedKeyword]
      secondStageDownCombined = [c for c in secondStageDownCombined if str(c) != 'nan']
      secondStageCombined = secondStageUpCombined + secondStageDownCombined

      metricsCEP, cmCEP, fprCEP, tprCEP, aucValuesCEP = CalculateMetrics(clsOriginal, secondStageCombined)
      metricsCEP["Mean_Score"] = np.array([metricsCEP[c] for c in columns2Print[2:]]).mean()
      metricsCEP["First_Stage_Keyword"] = firstStageCombinedKeyword
      metricsCEP["Second_Stage_Up_Keyword"] = secondStageUpCombinedKeyword
      metricsCEP["Second_Stage_Down_Keyword"] = secondStageDownCombinedKeyword
      metricsCEP["With Error Propagation"] = "Yes"

      secondStageNoCEP = []
      clsOriginalNoCEP = []
      for i in range(len(cls)):
        if (firstStageCombined[i] == cls[i]):
          secondStageNoCEP.append(secondStageCombined[i])
          clsOriginalNoCEP.append(clsOriginal[i])

      metricsNoCEP, cmNoCEP, fprNoCEP, tprNoCEP, aucValuesNoCEP = CalculateMetrics(clsOriginalNoCEP, secondStageNoCEP)
      metricsNoCEP["Mean_Score"] = np.array([metricsNoCEP[c] for c in columns2Print[2:]]).mean()
      metricsNoCEP["First_Stage_Keyword"] = firstStageCombinedKeyword
      metricsNoCEP["Second_Stage_Up_Keyword"] = secondStageUpCombinedKeyword
      metricsNoCEP["Second_Stage_Down_Keyword"] = secondStageDownCombinedKeyword
      metricsNoCEP["With Error Propagation"] = "No"

      finalResults = {
        "First_Stage_Single"   : metricsEP,
        "Second_Stage_Single"  : metricsNoEP,
        "First_Stage_Combined" : metricsCEP,
        "Second_Stage_Combined": metricsNoCEP,
      }

      finalResultsDF = pd.DataFrame(finalResults).T
      finalResultsDF.drop(["Metrics_Mean"], axis=1, inplace=True)
      # Round the values to 2 decimal points after multiplying by 100.
      finalResultsDF[columns2Print[2:] + ["Mean_Score"]] = finalResultsDF[columns2Print[2:] + ["Mean_Score"]] * 100
      finalResultsDF[columns2Print[2:] + ["Mean_Score"]] = finalResultsDF[columns2Print[2:] + ["Mean_Score"]].applymap(
        lambda x: f"{x:.2f}")
      finalResultsDF.to_csv(os.path.join(clsFolder, f"Final_Results.csv"), index=False)

      # Print LaTeX table.
      finalResultsDF = finalResultsDF[[
                                        "First_Stage_Keyword",
                                        "Second_Stage_Up_Keyword",
                                        "Second_Stage_Down_Keyword",
                                        "With Error Propagation"
                                      ] + columns2Print[2:] + ["Mean_Score"]]
      latexContent = finalResultsDF.style.hide_index().to_latex(
        position_float="centering",
        caption=r"First and Second Stage Performance Summary: Individual and Combined Classifier Metrics.",
        multirow_align="c",
        column_format="|c" * (finalResultsDF.shape[1] + 1) + "|",
        position="!htb",
        hrules=1,
        siunitx=True,
      )
      # print(latexContent)

      # Save LaTeX table.
      with open(os.path.join(latexFolder, f"Final_Results.tex"), "w") as f:
        f.write(latexContent)

      # Plot the four confusion matrices.
      currentFontSize = plt.rcParams["font.size"]
      plt.rcParams.update({"font.size": 18})  # Increase font size.
      fig, ax = plt.subplots(2, 2, figsize=(20, 20))
      disp = ConfusionMatrixDisplay(confusion_matrix=cmEP, display_labels=np.unique(clsOriginal))
      disp.plot(ax=ax[0, 0])
      ax[0, 0].set_title("With Error Propagation (Individual)", fontsize=20)
      disp = ConfusionMatrixDisplay(confusion_matrix=cmNoEP, display_labels=np.unique(clsOriginalNoEP))
      disp.plot(ax=ax[0, 1])
      ax[0, 1].set_title("Without Error Propagation (Individual)", fontsize=20)
      disp = ConfusionMatrixDisplay(confusion_matrix=cmCEP, display_labels=np.unique(clsOriginal))
      disp.plot(ax=ax[1, 0])
      ax[1, 0].set_title("With Error Propagation (Combined)", fontsize=20)
      disp = ConfusionMatrixDisplay(confusion_matrix=cmNoCEP, display_labels=np.unique(clsOriginalNoCEP))
      disp.plot(ax=ax[1, 1])
      ax[1, 1].set_title("Without Error Propagation (Combined)", fontsize=20)
      plt.tight_layout()
      plt.savefig(os.path.join(clsFolder, "ConfusionMatrices.pdf"), bbox_inches="tight", dpi=500)
      plt.close()
      gc.collect()
      plt.rcParams.update({"font.size": currentFontSize})  # Reset the font size.

      CL_PROGRESS += 1.5

      # ============================================================================================= #

      # ============================================================================================= #

      configurations = {
        "Features File"        : featuresFile,
        "Train to Test Ratio"  : trainToTestRatio,
        "Test Ratio"           : testRatio,
        "Outliers Detection"   : outliersDetection,
        "Outliers Fraction"    : outliersFraction,
        "Features Selection"   : featuresSelection,
        "Correlation Threshold": correlationThreshold,
        "Scalers"              : scaling,
        "Scalers List"         : scalersList,
        "Classifiers"          : classifiers,
        "Classifiers List"     : classifiersList,
        "Number of Trials"     : noOfTrials,
        "Columns To Drop"      : COLS_TO_DROP,
        "Random State"         : RANDOM_STATE,
        "Timestamp"            : CURRENT_TIMESTAMP,
        "Datetime"             : CURRENT_DATETIME,
        "Date"                 : CURRENT_DATE,
        "Current Date Time"    : currentDT,
      }

      configurationsPath = os.path.join(clsFolder, "Configurations.pickle")

      with open(configurationsPath, "wb") as f:
        pickle.dump(configurations, f)

      with open(configurationsPath, "rb") as f:
        configurations = pickle.load(f)

      # print(configurations)

      print("Features File:", configurations["Features File"])
      print("Train to Test Ratio:", configurations["Train to Test Ratio"])
      print("Test Ratio:", configurations["Test Ratio"])
      print("Outliers Detection:", "Yes" if configurations["Outliers Detection"] else "No")
      print("Outliers Fraction:", configurations["Outliers Fraction"])
      print("Features Selection:", "Yes" if configurations["Features Selection"] else "No")
      print("Correlation Threshold:", configurations["Correlation Threshold"])
      # print("Scalers:", configurations["Scalers"])
      print("Scalers List:", configurations["Scalers List"])
      # print("Classifiers:", configurations["Classifiers"])
      print("Classifiers List:", configurations["Classifiers List"])
      print("Number of Trials:", configurations["Number of Trials"])
      print("Columns To Drop:", configurations["Columns To Drop"])
      print("Random State:", configurations["Random State"])
      print("Timestamp:", configurations["Timestamp"])
      print("Datetime:", configurations["Datetime"])
      print("Date:", configurations["Date"])
      print("Current Date Time:", configurations["Current Date Time"])

      CL_PROGRESS = 100

      return jsonify(
        {
          "progress": CL_PROGRESS,
          "message" : "Task completed successfully.",
          "status"  : "success",
        }
      )
    except Exception as e:
      logger.info(f"[Classification Phase] Error: {str(ex)}.")
      # Log the ex line.
      logger.info(f"[Classification Phase] Error: {traceback.format_exc()}")

      CL_PROGRESS = 100
      return jsonify(
        {
          "progress": CL_PROGRESS,
          "message" : f"An error occurred during the classification process.",
          "status"  : "error",
        }
      )


  else:
    featuresPath = app.config["FEATURES_FOLDER"]
    featuresFolders = os.listdir(featuresPath)
    featuresFolders = [f for f in featuresFolders if os.path.isdir(os.path.join(featuresPath, f))]
    featuresFolders = sorted(featuresFolders)

    featuresContent = []
    for folder in featuresFolders:
      folderPath = os.path.join(featuresPath, folder, "Features.csv")
      configurationsFile = os.path.join(featuresPath, folder, "Configurations.pickle")
      if ((not os.path.exists(folderPath)) or (not os.path.exists(configurationsFile))):
        continue
      fileCreationDate = os.path.getctime(folderPath)
      fileCreationDateFormatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fileCreationDate))

      if (os.path.exists(folderPath)):
        featuresContent.append({
          "folder"          : folder,
          "fileCreationDate": fileCreationDateFormatted,
        })

    classificationImage = Image2Base64("static/images/Classification.jpg").decode("utf-8")

    return render_template(
      "classification.html",
      featuresContent=featuresContent,
      classificationImage=classificationImage,
    )


@app.route("/classification-navigation", methods=["GET"])
def classificationNavigation():
  classificationPath = app.config["CLASSIFICATION_FOLDER"]
  classificationImage = Image2Base64("static/images/Classification.jpg").decode("utf-8")

  folders = os.listdir(classificationPath)
  folders = [f for f in folders if os.path.isdir(os.path.join(classificationPath, f))]
  folders = sorted(folders)

  allFolders = {}
  for folder in folders:
    allFolders[folder] = {}
    innerFolders = os.listdir(os.path.join(classificationPath, folder))
    innerFolders = [f for f in innerFolders if os.path.isdir(os.path.join(classificationPath, folder, f))]
    innerFolders = sorted(innerFolders)
    for innerFolder in innerFolders:
      configPath = os.path.join(classificationPath, folder, innerFolder, "Configurations.pickle")
      if (not os.path.exists(configPath)):
        continue
      with open(configPath, "rb") as f:
        configurations = pickle.load(f)

      fst1im = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "First_Stage_Top1_Individual_Metrics.csv"))
      ssut1im = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "Second_Stage_Up_Top1_Individual_Metrics.csv"))
      ssdt1im = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "Second_Stage_Down_Top1_Individual_Metrics.csv"))

      fst1t1cm = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "First_Stage_Top1_Top1_Combinations_Metrics.csv"))
      ssut1t1cm = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "Second_Stage_Up_Top1_Top1_Combinations_Metrics.csv"))
      ssdt1t1cm = pd.read_csv(
        os.path.join(classificationPath, folder, innerFolder, "Second_Stage_Down_Top1_Top1_Combinations_Metrics.csv"))

      fr = pd.read_csv(os.path.join(classificationPath, folder, innerFolder, "Final_Results.csv"))
      frCols = list(fr.columns)
      frCols = frCols[17:] + frCols[:10] + ["AUC_Mean"] + ["Mean_Score"] + frCols[10:15]
      fr = fr[frCols]

      frLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results", "Final_Results.tex")
      if (os.path.exists(frLaTeXFile)):
        with open(frLaTeXFile, "r") as f:
          frLaTeX = f.read()
      else:
        frLaTeX = ""
      frLaTeX = frLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

      fst1imLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                     "First_Stage_Top1_Individual_Metrics.tex")
      if (os.path.exists(fst1imLaTeXFile)):
        with open(fst1imLaTeXFile, "r") as f:
          fst1imLaTeX = f.read()
      else:
        fst1imLaTeX = ""
      fst1imLaTeX = fst1imLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

      ssut1imLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                      "Second_Stage_Up_Top1_Individual_Metrics.tex")
      if (os.path.exists(ssut1imLaTeXFile)):
        with open(ssut1imLaTeXFile, "r") as f:
          ssut1imLaTeX = f.read()
      else:
        ssut1imLaTeX = ""
      ssut1imLaTeX = ssut1imLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

      ssdt1imLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                      "Second_Stage_Down_Top1_Individual_Metrics.tex")
      if (os.path.exists(ssdt1imLaTeXFile)):
        with open(ssdt1imLaTeXFile, "r") as f:
          ssdt1imLaTeX = f.read()
      else:
        ssdt1imLaTeX = ""
      ssdt1imLaTeX = ssdt1imLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

      fst1t1cmLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                       "First_Stage_Top1_Top1_Combinations_Metrics.tex")
      if (os.path.exists(fst1t1cmLaTeXFile)):
        with open(fst1t1cmLaTeXFile, "r") as f:
          fst1t1cmLaTeX = f.read()
      else:
        fst1t1cmLaTeX = ""
      fst1t1cmLaTeX = fst1t1cmLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t",
                                                                                         "&nbsp;&nbsp;&nbsp;&nbsp;")

      ssut1t1cmLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                        "Second_Stage_Up_Top1_Top1_Combinations_Metrics.tex")
      if (os.path.exists(ssut1t1cmLaTeXFile)):
        with open(ssut1t1cmLaTeXFile, "r") as f:
          ssut1t1cmLaTeX = f.read()
      else:
        ssut1t1cmLaTeX = ""
      ssut1t1cmLaTeX = ssut1t1cmLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t",
                                                                                           "&nbsp;&nbsp;&nbsp;&nbsp;")

      ssdt1t1cmLaTeXFile = os.path.join(classificationPath, folder, innerFolder, "LaTeX Results",
                                        "Second_Stage_Down_Top1_Top1_Combinations_Metrics.tex")
      if (os.path.exists(ssdt1t1cmLaTeXFile)):
        with open(ssdt1t1cmLaTeXFile, "r") as f:
          ssdt1t1cmLaTeX = f.read()
      else:
        ssdt1t1cmLaTeX = ""
      ssdt1t1cmLaTeX = ssdt1t1cmLaTeX.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t",
                                                                                           "&nbsp;&nbsp;&nbsp;&nbsp;")

      allFolders[folder][innerFolder] = {
        "Configurations"                                : {
          "Features File"        : configurations["Features File"],
          "Train to Test Ratio"  : configurations["Train to Test Ratio"] + "%",
          # "Test Ratio"           : configurations["Test Ratio"],
          "Outliers Detection"   : "Yes" if (configurations["Outliers Detection"] == "1") else "No",
          "Outliers Fraction"    : configurations["Outliers Fraction"] + "%" if (
            configurations["Outliers Detection"] == "1") else "N/A",
          "Features Selection"   : "Yes" if (configurations["Features Selection"] == "1") else "No",
          "Correlation Threshold": configurations["Correlation Threshold"] + "%" if (
            configurations["Features Selection"] == "1") else "N/A",
          "Scalers"              : configurations["Scalers"],
          # "Scalers List"         : configurations["Scalers List"],
          "Classifiers"          : configurations["Classifiers"],
          # "Classifiers List"     : configurations["Classifiers List"],
          "Number of Trials"     : configurations["Number of Trials"],
        },
        "First_Stage_Individual_Metrics"                : fst1im.values.tolist(),
        "First_Stage_Individual_Metrics_Columns"        : list(fst1im.columns),
        "First_Stage_Individual_Metrics_LaTeX"          : fst1imLaTeX,
        "Second_Stage_Up_Individual_Metrics"            : ssut1im.values.tolist(),
        "Second_Stage_Up_Individual_Metrics_Columns"    : list(ssut1im.columns),
        "Second_Stage_Up_Individual_Metrics_LaTeX"      : ssut1imLaTeX,
        "Second_Stage_Down_Individual_Metrics"          : ssdt1im.values.tolist(),
        "Second_Stage_Down_Individual_Metrics_Columns"  : list(ssdt1im.columns),
        "Second_Stage_Down_Individual_Metrics_LaTeX"    : ssdt1imLaTeX,
        "First_Stage_Combinations_Metrics"              : fst1t1cm.values.tolist(),
        "First_Stage_Combinations_Metrics_Columns"      : list(fst1t1cm.columns),
        "First_Stage_Combinations_Metrics_LaTeX"        : fst1t1cmLaTeX,
        "Second_Stage_Up_Combinations_Metrics"          : ssut1t1cm.values.tolist(),
        "Second_Stage_Up_Combinations_Metrics_Columns"  : list(ssut1t1cm.columns),
        "Second_Stage_Up_Combinations_Metrics_LaTeX"    : ssut1t1cmLaTeX,
        "Second_Stage_Down_Combinations_Metrics"        : ssdt1t1cm.values.tolist(),
        "Second_Stage_Down_Combinations_Metrics_Columns": list(ssdt1t1cm.columns),
        "Second_Stage_Down_Combinations_Metrics_LaTeX"  : ssdt1t1cmLaTeX,
        "Final_Results"                                 : fr.values.tolist(),
        "Final_Results_Columns"                         : list(fr.columns),
        "Final_Results_LaTeX"                           : frLaTeX,
      }

  return render_template(
    "classification_navigation.html",
    classificationImage=classificationImage,
    allFolders=allFolders,
  )


@app.route("/inference", methods=["GET", "POST"])
def inference():
  global IN_PROGRESS, ALLOWED_EXTENSIONS
  classificationPath = app.config["CLASSIFICATION_FOLDER"]
  featuresPath = app.config["FEATURES_FOLDER"]

  if (request.method == "POST"):
    subFolder = request.form["subFolder"]

    if (len(subFolder) == 0):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": f"Experiment is required.",
          "status" : "error",
        }
      )

    subFolderPath = os.path.join(classificationPath, subFolder)
    if (not os.path.exists(subFolderPath)):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": f"Experiment ({subFolder}) does not exist.",
          "status" : "error",
        }
      )

    if ("image" not in request.files) or ("mask" not in request.files):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": "Image and mask files are required.",
          "status" : "error",
        }
      )

    image = request.files["image"]
    mask = request.files["mask"]

    if (image.filename == ""):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": "Image file is required.",
          "status" : "error",
        }
      )

    if (mask.filename == ""):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": "Mask file is required.",
          "status" : "error",
        }
      )

    # Check if not allowed file type.
    imgExtension = image.filename.split(".")[-1]
    maskExtension = mask.filename.split(".")[-1]
    if (imgExtension not in ALLOWED_EXTENSIONS) or (maskExtension not in ALLOWED_EXTENSIONS):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": "Only JPG, JPEG, PNG, and BMP files are allowed.",
          "status" : "error",
        }
      )

    IN_PROGRESS = 10

    predictionsFolder = os.path.join(subFolderPath, "Predictions")
    os.makedirs(predictionsFolder, exist_ok=True)
    timestamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(CURRENT_TIMESTAMP))}"
    imgPath = os.path.join(predictionsFolder, f"{timestamp}_Image.jpg")
    maskPath = os.path.join(predictionsFolder, f"{timestamp}_Mask.jpg")
    image.save(imgPath)
    mask.save(maskPath)

    IN_PROGRESS = 20

    featuresFolderPath = os.path.join(featuresPath, subFolder.split("\\")[0])
    resultsDict = InferencePhase(subFolderPath, imgPath, maskPath, featuresFolderPath, timestamp)
    if (resultsDict is None):
      IN_PROGRESS = 100
      return jsonify(
        {
          "message": "An error occurred during the inference process.",
          "status" : "error",
        }
      )
    plot = resultsDict["PlotPath"]
    plotImg = Image2Base64(plot).decode("utf-8")
    outputLabel = resultsDict["Second_Stage_Final_Prediction_Label"]
    imgEnc = Image2Base64(imgPath).decode("utf-8")
    maskEnc = Image2Base64(maskPath).decode("utf-8")

    IN_PROGRESS = 100

    return jsonify(
      {
        "message"    : "Prediction completed successfully.",
        "status"     : "success",
        "plot"       : plotImg,
        "outputLabel": outputLabel,
        "image"      : imgEnc,
        "mask"       : maskEnc,
      }
    )

  else:
    folders = os.listdir(classificationPath)
    subFolders = []
    for folder in folders:
      subFolders.extend([
        os.path.join(folder, f)
        for f in os.listdir(os.path.join(classificationPath, folder))
      ])

    return render_template(
      "inference.html",
      subFolders=subFolders,
    )


@app.route("/about", methods=["GET"])
def about():
  bonesImage = Image2Base64("static/images/Bones.jpg").decode("utf-8")
  frameworkImage = Image2Base64("static/images/Framework.jpg").decode("utf-8")
  cmImage = Image2Base64("static/images/ConfusionMatrices.jpg").decode("utf-8")
  return render_template(
    "about.html",
    bonesImage=bonesImage,
    frameworkImage=frameworkImage,
    cmImage=cmImage,
  )


@app.route("/contact", methods=["GET"])
def contact():
  return render_template("contact.html")


if (__name__ == "__main__"):
  app.run(
    debug=True,
    threaded=True,
    port=5000,
    host="",
    # use_reloader=False,
    # use_evalex=False,
    # use_debugger=False,
  )
