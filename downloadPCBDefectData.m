function downloadPCBDefectData(dataDir)
% downloadPCBDefectData downloads the PCB data set.

% Copyright 2022 The MathWorks, Inc.

if ~exist(dataDir,"dir")   
    mkdir(dataDir);
end

imageDir = fullfile(dataDir,"PCB-DATASET-master");

if ~exist(imageDir,"dir")
    disp("Downloading PCB data set.");
    disp("This can take several minutes to download and unzip...");
    dataURL = "https://github.com/Ironbrotherstyle/PCB-DATASET/archive/refs/heads/master.zip";
    unzip(dataURL,dataDir);
    delete(fullfile(imageDir,"*.m"),fullfile(imageDir,"*.mlx"), ...
        fullfile(imageDir,"*.mat"),fullfile(imageDir,"*.md")); 
    disp("Done.");
end

end
