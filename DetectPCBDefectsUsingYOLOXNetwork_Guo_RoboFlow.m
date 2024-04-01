%% Download pretrained YOLOX detector
%trainedPCBDefectDetectorNet_url = "https://ssd.mathworks.com/supportfiles/"+ ...
 %   "vision/data/trainedPCBDefectDetectorYOLOX.zip";
%downloadTrainedNetwork(trainedPCBDefectDetectorNet_url,pwd);
load("trainedPCBDefectDetectorYOLOX.mat");

%% Example detection and classification on one image
%sampleImage = imread(fullfile("Zhu (Roboflow)", "images", ...
%    "01_missing_hole_01_jpg.rf.03d45aa38c40a08fbc3a01bfb1dfa434.jpg"));
%imshow(sampleImage);
%[bboxes,scores,labels] = detect(detector,sampleImage);
%title("Predicted Defects");
%showShape("rectangle", bboxes,Label=labels);
%ax = gcf;
%exportgraphics(ax, "YOLOX predictions (Zhu Roboflow).jpg");

%% Prepare data for (hypothetical) training
% Create an image datastore that reads and manages the image data.
imageDir = fullfile("Guo (Roboflow)", "images");
imds = imageDatastore(imageDir,FileExtensions=[".jpg", ".png"],IncludeSubfolders=true);
%Create a file datastore that reads the annotation data from XML files. 
% Specify a custom read function that parses the XML files and extracts the bounding box information. 
% The custom read function, readPCBDefectAnnotations, is attached to the example as a supporting file.
annoDir = fullfile("Guo (Roboflow)", "annotations");
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
rng("default");
detectionResults = detect(detector, ds);
save('guo_beta_1dot0_roboflow_raw_detection.mat', 'detectionResults');

%% Analyze Object Class Distribution
% Measure the distribution of class labels in the data set by using the countEachLabel function
countEachLabelTable = countEachLabel(blds);
writetable(countEachLabelTable, "countEachLabelTable.csv");

%% Partition Data

% Split the data set into training, validation, and test sets. 
% Because the total number of images is relatively small, 
% allocate a relatively large percentage (70%) of the data for training. 
% Allocate 15% for validation and the rest for testing.
%numImages = ds.numpartitions;
%numTrain = floor(0.7*numImages);
%numVal = floor(0.15*numImages);

%shuffledIndices = randperm(numImages);
%dsTrain = subset(ds,shuffledIndices(1:numTrain));
%dsVal = subset(ds,shuffledIndices(numTrain+1:numTrain+numVal));
%dsTest = subset(ds,shuffledIndices(numTrain+numVal+1:end));
%dsTest = subset(ds,shuffledIndices(1:end));

%% Evaluate Detector
% Evaluate the trained object detector by measuring the average precision. 
% Precision quantifies the ability of the detector to classify objects correctly.

%Detect the bounding boxes for all test images.


% Create a directory for saving figures if it doesn't exist
if ~exist('detection_figures_guo_roboflow_figures', 'dir')
    mkdir('detection_figures_guo_roboflow_figures');
end

% Create a directory for saving figures if it doesn't exist
if ~exist('detection_figures_guo_roboflow_images', 'dir')
    mkdir('detection_figures_guo_roboflow_images');
end

% Iterate over detection results
for i = 1:size(detectionResults, 1)
    % Get bounding boxes, scores, and labels for current detection
    bboxes_cell = detectionResults.Boxes(i, :);
    scores_cell = detectionResults.Scores(i, :);
    labels_cell = detectionResults.Labels(i, :);
    
    bboxes = bboxes_cell{:};
    scores = scores_cell{:};
    labels = labels_cell{:};

    % Get the filename of the original image
    [~, filename, ext] = fileparts(ds.UnderlyingDatastores{1,1}.Files{i});
    
    % Read the original image
    img = imread(ds.UnderlyingDatastores{1,1}.Files{i, 1});
    
    % Create a new figure
    fig = figure('Visible', 'off');
    imshow(img);
    title("Predicted Defects");
    showShape("rectangle",bboxes,Label=labels);
    ax = gcf;

    % Save the figure as .fig
    filename = strrep(filename, '.', ''); % Remove dots from filename
    %fig_filename = fullfile('detection_figures_guo_roboflow_figures', [filename, '.fig']);
    %savefig(ax, fig_filename);
    
    % Save the figure as .m (using saveas)
    %m_filename = fullfile('detection_figures_guo_roboflow_figures', [filename, '.m']);
    %saveas(ax, m_filename);
    
    % Save the figure as .png
    png_filename = fullfile('detection_figures_guo_roboflow_images', [filename, '.png']);
    exportgraphics(ax, png_filename);
    
    % Close the figure
    close(fig);    
end


% Calculate the average precision score for each class by using the evaluateObjectDetection function. 
% Also calculate the recall and precision values for the detection of each defect object. 
% Recall quantifies the ability of the detector to detect all relevant objects for a class.
metrics = evaluateObjectDetection(detectionResults, ds, AdditionalMetrics=["LAMR"]);

% Save total metrics to a .mat file
save('guo_beta_1dot0_roboflow_metrics.mat', 'metrics');
precision = metrics.ClassMetrics.Precision;
recall = metrics.ClassMetrics.Recall;
averagePrecision = cell2mat(metrics.ClassMetrics.AP);

% Record the average precision score for each class.
classNames = replace(classNames,"_"," ");
averagePrecisionTable = table(classNames,averagePrecision);
writetable(averagePrecisionTable, "AveragePrecisionTable.csv");

% A precision-recall (PR) curve highlights how precise a detector is at varying levels of recall. 
% The ideal precision is 1 at all recall levels. Plot the PR curve for the test data.
for class = 1:6
    plot(recall{class}, precision{class});
    xlabel("Recall")
    ylabel("Precision")
    title(sprintf("Average Precision for '" + classNames(class) + "' Defect: " + "%.2f", averagePrecision(class)))
    grid on
    ax = gca;
    saveas(ax, "PR Curve for " + classNames(class) + ".m");
    exportgraphics(ax, "PR Curve for " + classNames(class) + ".jpg");
    % Add pause if you want to see each plot separately
    % pause(2); 
end

%% Evaluate Object Size-based Detection Metrics
% Investigate the impact of object size on detector performance with the metricsByArea function, 
% which computes the object detection metrics for specific object size ranges. 
% To evaluate size-based metrics, you can define the object size ranges based on a custom set of ranges. 
% First, break the test image bounding box sizes into small, medium, and large object size categories
% according to percentile boundaries of the 33rd and 66th percentile of the test set object area distribution. 

%Plot the test set object size distribution, where the bounding box area defines the object size.
testSetObjects = ds.UnderlyingDatastores{2};
objectLabels = readall(testSetObjects);
boxes = objectLabels(:,1);
boxes = vertcat(boxes{:});
boxArea = prod(boxes(:,3:4),2);
histogram(boxArea);
xlabel("Box Area")
ylabel("Count")
title("Bounding Box Area Distribution")
ax = gca;
saveas(ax, "Bounding Box Area Distribution.m");
exportgraphics(ax, "Bounding Box Area Distribution.jpg");

% Define the bounding box area ranges, and then evaluate object detection metrics 
% for the defined area ranges using metricsByArea. 
% The mean average precision (mAP) metric of the trained detector performs 
% approximately the same across small, medium, and large object sizes, 
% with a slight performance improvement for medium objects.
boxPrctileBoundaries = prctile(boxArea,100*[1/3,2/3]);
metricsByAreaTable = metricsByArea(metrics,[0, boxPrctileBoundaries, inf]);
writetable(metricsByAreaTable, "metricsByAreaTable.csv");