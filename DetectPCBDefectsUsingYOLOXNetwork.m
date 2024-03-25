%% Download pretrained YOLOX detector
trainedPCBDefectDetectorNet_url = "https://ssd.mathworks.com/supportfiles/"+ ...
    "vision/data/trainedPCBDefectDetectorYOLOX.zip";
downloadTrainedNetwork(trainedPCBDefectDetectorNet_url,pwd);
load("trainedPCBDefectDetectorYOLOX.mat");

%% Example detection and classification on one image
sampleImage = imread(fullfile("pcb_defect_detection_v2i_voc", "images", ...
    "01_missing_hole_01_jpg.rf.03d45aa38c40a08fbc3a01bfb1dfa434.jpg"));
%sampleImage = cat(3, sampleImage, sampleImage, sampleImage);
imshow(sampleImage);
[bboxes,scores,labels] = detect(detector,sampleImage);
title("Predicted Defects");
showShape("rectangle",bboxes,Label=labels);
ax = gcf;
exportgraphics(ax, "YOLOX predictions.jpg");

%% Prepare data for (hypothetical) training
% Create an image datastore that reads and manages the image data.
imageDir = fullfile("pcb_defect_detection_v2i_voc","images");
imds = imageDatastore(imageDir,FileExtensions=".jpg",IncludeSubfolders=true);
%Create a file datastore that reads the annotation data from XML files. 
% Specify a custom read function that parses the XML files and extracts the bounding box information. 
% The custom read function, readPCBDefectAnnotations, is attached to the example as a supporting file.
annoDir = fullfile("pcb_defect_detection_v2i_voc","annotations");
fds = fileDatastore(annoDir,ReadFcn=@readPCBDefectAnnotations, ...
    FileExtensions=".xml",IncludeSubfolders=true);
% Save the labeled bounding box data as a box label datastore.
annotations = readall(fds);
tbl = struct2table(vertcat(annotations{:}));
blds = boxLabelDatastore(tbl);
% Get the names of the object classes as a categorical vector.
classNames = categories(blds.LabelData{1,2});
% Combine the image and box label datastores.
ds = combine(imds,blds);

%% Analyze Object Class Distribution
% Measure the distribution of class labels in the data set by using the countEachLabel function
countEachLabelTable = countEachLabel(blds);
writetable(countEachLabelTable, "countEachLabelTable.csv");

%% Partition Data
rng("default");
% Split the data set into training, validation, and test sets. 
% Because the total number of images is relatively small, 
% allocate a relatively large percentage (70%) of the data for training. 
% Allocate 15% for validation and the rest for testing.
numImages = ds.numpartitions;
%numTrain = floor(0.7*numImages);
%numVal = floor(0.15*numImages);

shuffledIndices = randperm(numImages);
%dsTrain = subset(ds,shuffledIndices(1:numTrain));
%dsVal = subset(ds,shuffledIndices(numTrain+1:numTrain+numVal));
%dsTest = subset(ds,shuffledIndices(numTrain+numVal+1:end));
dsTest = subset(ds,shuffledIndices(1:end));

%% Evaluate Detector
% Evaluate the trained object detector by measuring the average precision. 
% Precision quantifies the ability of the detector to classify objects correctly.

%Detect the bounding boxes for all test images.
detectionResults = detect(detector,dsTest);

% Calculate the average precision score for each class by using the evaluateObjectDetection function. 
% Also calculate the recall and precision values for the detection of each defect object. 
% Recall quantifies the ability of the detector to detect all relevant objects for a class.
metrics = evaluateObjectDetection(detectionResults,dsTest);
precision = metrics.ClassMetrics.Precision;
recall = metrics.ClassMetrics.Recall;
averagePrecision = cell2mat(metrics.ClassMetrics.AP);

% Record the average precision score for each class.
classNames = replace(classNames,"_"," ");
averagePrecisionTable = table(classNames,averagePrecision);
writetable(averagePrecisionTable, "AveragePrecisionTable.csv");

% A precision-recall (PR) curve highlights how precise a detector is at varying levels of recall. 
% The ideal precision is 1 at all recall levels. Plot the PR curve for the test data.
class = 1;
plot(recall{class},precision{class});
xlabel("Recall");
ylabel("Precision");
title(sprintf("Average Precision for '" + classNames(class) + "' Defect: " + "%.2f",averagePrecision(class)));
grid on;
ax = gca;
exportgraphics(ax, "Average Precision for " + classNames(class) + ".jpg");

%% Evaluate Object Size-based Detection Metrics
% Investigate the impact of object size on detector performance with the metricsByArea function, 
% which computes the object detection metrics for specific object size ranges. 
% To evaluate size-based metrics, you can define the object size ranges based on a custom set of ranges. 
% First, break the test image bounding box sizes into small, medium, and large object size categories
% according to percentile boundaries of the 33rd and 66th percentile of the test set object area distribution. 

%Plot the test set object size distribution, where the bounding box area defines the object size.
testSetObjects = dsTest.UnderlyingDatastores{2};
objectLabels = readall(testSetObjects);
boxes = objectLabels(:,1);
boxes = vertcat(boxes{:});
boxArea = prod(boxes(:,3:4),2);
histogram(boxArea);
xlabel("Box Area");
ylabel("Count");
title("Bounding Box Area Distribution");
ax = gca;
exportgraphics(ax, "Bounding Box Area Distribution.jpg");

% Define the bounding box area ranges, and then evaluate object detection metrics 
% for the defined area ranges using metricsByArea. 
% The mean average precision (mAP) metric of the trained detector performs 
% approximately the same across small, medium, and large object sizes, 
% with a slight performance improvement for medium objects.
boxPrctileBoundaries = prctile(boxArea,100*[1/3,2/3]);
metricsByAreaTable = metricsByArea(metrics,[0, boxPrctileBoundaries, inf]);
writetable(metricsByAreaTable, "metricsByAreaTable.csv");