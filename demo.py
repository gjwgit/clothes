# -*- coding: utf-8 -*-

print("Loading the required libraries...")

from helpers import *
from helpers_cntk import *
import os
from sys import platform
locals().update(importlib.import_module("PARAMETERS").__dict__)
from random import sample
import subprocess

####################################
# Specify Input Image Data for Scoring
####################################

# makeDirectory(rootDir + "/data/")
# makeDirectory(imgDir)
imgDir = "data/fashionTexture/"

####################################
# Prepare Data
####################################

random.seed(0)
makeDirectory(procDir)
imgFilenamesTest  = dict()
imgFilenamesTrain = dict()

print("""
We will now prepare the sample images found in '{}'""".format(imgDir))

subdirs = getDirectoriesInDirectory(imgDir)
for subdir in subdirs:
    filenames = getFilesInDirectory(imgDir + subdir, ".jpg")
    
    # Assign images into test
    if imagesSplitBy == 'filename':
        filenames = randomizeList(filenames)
        splitIndex = int(0 * len(filenames))
        imgFilenamesTrain[subdir] = filenames[:splitIndex]
        imgFilenamesTest[subdir] = filenames[splitIndex:]
        
    # Assign whole subdirectories to test
    elif imagesSplitBy == 'subdir':
        if random.random() < 0:
            imgFilenamesTrain[subdir] = filenames
        else:
            imgFilenamesTest[subdir] = filenames
    else:
        raise Exception("Variable 'imagesSplitBy' has to be either 'filename' or 'subdir'")

    # Debug print
    if subdir in imgFilenamesTest:
        print(""" {:2} images in the sub-directory '{}'""".format(len(imgFilenamesTest[subdir]), subdir))

# Save assignments of images to test

saveToPickle(imgFilenamesTestPath,  imgFilenamesTest)

# Compute positive and negative image pairs

print("""
Computing image pairs from the sample dataset...""")
imgInfosTest = getImagePairs(imgFilenamesTest, test_maxQueryImgsPerSubdir, test_maxNegImgsPerQueryImg)
saveToPickle(imgInfosTestPath, imgInfosTest)

sys.stderr.write("""
Data preparation is now complete. Please press Enter to continue: """)
input()
print("")

################################################
# Featurize Images
################################################
# Init
printDeviceType()
makeDirectory(workingDir)
model = load_model(cntkRefinedModelPath)

# Compute features for each image and write to disk.

print("\nWe will now featurize the dataset of images...")
featuresTest  = featurizeImages(model, imgFilenamesTestPath,  imgDir, workingDir + "/featurizer_map.txt", "poolingLayer", run_mbsize)
features = featuresTest
for feat in list(features.values()):
    assert(len(feat) == rf_modelOutputDimension)

# Save features to file
print("\nCNTK outputs will be saved into\n %s ..." % featuresPath)
saveToPickle(featuresPath, features)

####################################
# Score SVM
####################################

# Parameter

distMethods = ['random', 'L1', 'L2', 'weighted'+svm_featureDifferenceMetric]  #'cosine', 'correlation', 'chiSquared', 'normalizedChiSquared']

# No need to change below parameters
boVisualizeResults  = True
boEvalOnTrainingSet = False  # Set to 'False' to evaluate using test set; 'True' to instead eval on training set
visualizationDir = resultsDir + "visualizations_weightedl2/" + "demo_results/"

random.seed(0)

# Load trained svm
learner    = loadFromPickle(svmPath)
svmBias    = learner.base_estimator.intercept_
svmWeights = np.array(learner.base_estimator.coef_[0])

# Load data
#print("[InProgress] Loading featurized test data...")
ImageInfo.allFeatures = loadFromPickle(featuresPath)
imgInfos = loadFromPickle(imgInfosTestPath)

# Compute distances between all image pairs
#print("[InProgress] Computing pair-wise distances...")
allDists = { queryIndex:collections.defaultdict(list) for queryIndex in range(len(imgInfos)) }
for queryIndex, queryImgInfo in enumerate(imgInfos):
    queryFeat = queryImgInfo.getFeat()
#    if queryIndex % 50 == 0:
        # print("Computing distances for query image {} of {}: {}..".format(queryIndex, len(imgInfos), queryImgInfo.fname))

    # Loop over all reference images and compute distances
    for refImgInfo in queryImgInfo.children:
        refFeat = refImgInfo.getFeat()
        for distMethod in distMethods:
            dist = computeVectorDistance(queryFeat, refFeat, distMethod, svm_boL2Normalize, svmWeights, svmBias)
            allDists[queryIndex][distMethod].append(dist)

# Find match with minimum distance (rank 1)
#print("[InProgress] Showing matching image and correct image with minimum weightedl2 distance for query image: ")

print("""
Display a selection of source images and minimum distance matches:""")

selected = random.sample(range(len(imgInfos)), 10)

fmt = "\n{0:<6.6} : {1:<7.7} : {2:<6.6} : {3:<7.7} : {4:<4.4} : {5:<6.6} : {6:<7.7} : {7:<4.4} : {8:<5.5}"
print(fmt.format("Img", "Label", "Match", "Label", "Dist", "Best", "Label", "Dist", "Error"))

fmt = "{0:>6.6} : {1:<7.7} : {2:>6.6} : {3:<7.7} : {4:>4.1f} : {5:>6.6} : {6:<7.7} : {7:>4.1f} : {8:<4.4}"
i = 0
for queryIndex, queryImgInfo in enumerate(imgInfos):
    i += 1
    dists = allDists[queryIndex]["weightedl2"]
    # Find match with minimum distance (rank 1)
    sortOrder = np.argsort(dists)
    minDistIndex = sortOrder[0]
    correctIndex = np.where([child.isSameClassAsParent() for child in queryImgInfo.children])[0][0]
    minDist      = dists[minDistIndex]
    correctDist  = dists[correctIndex]
    queryImgName     = queryImgInfo.fname
    minDistImgName   = imgInfos[queryIndex].children[minDistIndex].fname
    correctImgName   = imgInfos[queryIndex].children[correctIndex].fname
    queryLabel = queryImgInfo.subdir
    minDistLabel = imgInfos[queryIndex].children[minDistIndex].subdir    
    correctLabel = queryLabel
    if (minDistImgName == correctImgName):
        error = ""
    else:
        error = "<==="
    if (i in selected):
      print(fmt.format(queryImgName, queryLabel, minDistImgName, minDistLabel, minDist, correctImgName, correctLabel, correctDist, error))

# Check whether display is available
displayAvailable = os.path.exists("display.py")

if (displayAvailable):

    # Visualize
    if boVisualizeResults:
        makeDirectory(resultsDir)
        makeDirectory(visualizationDir)
        print("\nDemo images being written to:\n " +  visualizationDir)

        # Loop over all query images
        for queryIndex, queryImgInfo in enumerate(imgInfos):
            # print("   Visualizing result for query image: " + imgDir + queryImgInfo.fname)
            dists = allDists[queryIndex]["weightedl2"]

            # Find match with minimum distance (rank 1) and correct match
            sortOrder = np.argsort(dists)
            minDistIndex = sortOrder[0]
            correctIndex = np.where([child.isSameClassAsParent() for child in queryImgInfo.children])[0][0]
            minDist      = dists[minDistIndex]
            correctDist  = dists[correctIndex]
            queryImg     = queryImgInfo.getImg(imgDir)
            minDistImg   = imgInfos[queryIndex].children[minDistIndex].getImg(imgDir)
            correctImg   = imgInfos[queryIndex].children[correctIndex].getImg(imgDir)
            minDistLabel = imgInfos[queryIndex].children[minDistIndex].subdir

            # Visualize
            if minDistLabel == queryImgInfo.subdir:
                plt.rcParams['figure.facecolor'] = 'green' #correct ranking result
            else:
                plt.rcParams['figure.facecolor'] = 'red'
            pltAxes = [plt.subplot(1, 3, i+1) for i in range(3)]
            for ax, img, title in zip(pltAxes, (queryImg,minDistImg,correctImg),
                                  ('Query image', 'MinDist match \n (dist={:3.2f})'.format(minDist), 'Correct match \n (dist={:3.2f})'.format(correctDist))):
                ax.imshow(imconvertCv2Numpy(img))
                ax.axis('off')
                ax.set_title(title)
            plt.draw()
            plt.savefig(visualizationDir + "/" + queryImgInfo.fname.replace('/','-'), dpi=200, bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])

    if platform == "linux" or platform == "linux2":
        fn = ""
    else:
        fn = queryImgInfo.fname.replace('/','-')
        
    sys.stdout.write("""
Images have been saved. Please press Enter to continue on to view them: """)
    input()

    visualizationDir = resultsDir + "visualizations_weightedl2/" + "demo" + "_results/"

    if (platform == "linux" or platform == "linux2"):
        try:
            subprocess.call(["eom", visualizationDir])
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                image = Image.open(visualizationDir)
                image.show()
    else:
        image = Image.open(visualizationDir + "demo")
        image.show()

else:
    print("")
