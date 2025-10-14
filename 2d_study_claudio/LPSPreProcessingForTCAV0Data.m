addpath '/Users/cemma/Documents/Work/FACET-II/DAQ_and_Programming/DAQHelperFunctions'
addpath '/Users/cemma/Documents/Work/FACET-II/Laser_work/Laser_heater/LaserHeaterCommissioningShifts/DelayStageScans08142023/MatlabScriptScans'
addpath '/Users/cemma/Documents/Work/FACET-II/Laser_work/Laser_heater/LaserHeaterCommissioningShifts/COTRScansPR11375_09012023'

opts = struct('usemethod',2);% To use built-in stats Asym Gauss

% Load DAQ filenames and calibration info
lhDataSets(1).DAQPaths = {'/Users/cemma/nas/nas-li20-pm00/TEST/2023/20230901/TEST_03748'};

% Load images from DAQ and DAQ BSA scalars

    % Load DAQ .mat file        
    DAQpath =  lhDataSets(1).DAQPaths{1};
    load([DAQpath,'/',DAQpath(51:end),'.mat'])    

    screenRes = data_struct.metadata.PR10711.RESOLUTION;

    % Initialize image arrays to import DAQ images
    imgLoc = ['/Users/cemma/',data_struct.images.PR10711.loc{1}];
    [mm,nn]=size(imread(imgLoc));

    % Find matching timestamps between the cameras
    C =  data_struct.images.PR10711.common_index;

    % Load DAQ BSA scalar Data Nvars x Nshots
    [bsaScalarData,bsaVars] = extractDAQBSAScalars(data_struct);

    if size(bsaScalarData,2)~=length(C)
       disp('error')
        return
    end
    disp(['Length C = ',num2str(length(C))])
    if isempty(C);return;end

    % initialize dummy variables
    imgLoc = ['/Users/cemma/',data_struct.images.PR10711.loc{C(end)}];
    sampleImage = imrotate(imread(imgLoc),180);   
    sampleImgProc =  processNoisyTCAVImage(sampleImage,20,1,5);%process the image
        figure
    subplot(2,1,1)
    imagesc(sampleImage);colormap jetvar
    subplot(2,1,2)
    imagesc(sampleImgProc);colormap jetvar
%%
    % Loop over all shots

         for n = 1:length(C)
              
            imgLoc = ['/Users/cemma/',data_struct.images.PR10711.loc{C(n)}];
            imgfull(:,:) = imrotate(imread(imgLoc),180);    
            img = imgfull(1:10:491,50:300);%downsample and crop the image    
            imgProc =  processNoisyTCAVImage(img,20,1,5);%process the image
            %figure;subplot(1,2,1);imagesc(imgfull(:,:));subplot(1,2,2);imagesc(imgProc)
            lpsFlattened(n,:) = reshape(imgProc,1,[]);
            n
         end
% If you want to un-reshape the flattened LPS vectors you can do so 
         lpsRetrieved(ij,:,:) = reshape(lpsFlattened(n,:),size(imgProc,1),size(imgProc,2));
         figure;imagesc(squeeze(lpsRetrieved(ij,:,:)))

% Save the data so you can do an LPS prediction with a neural net model
 save('lpsFlattened_TEST_03748','lpsFlattened')
