% Chirp Calculator
% Calculates the chirp
addpath ./MATLAB_Functions  % For jetvar colormap

% Clears workspace
clear all
close all
clc

% Sets data location
experiment = 'E338';
runname = '12710';
date = '20250508';

% Define XTCAV calibration
krf = 239.26;
cal = 1167;% um/deg  http://physics-elog.slac.stanford.edu/facetelog/show.jsp?dir=/2025/11/13.03&pos=2025-03-13T07:16:12
streakFromGUI = cal*krf*180/pi*1e-6;%um/um

% Sets the main beam energy
mainbeamE_eV = 10e9;
% Sets the dnom value for CHER
dnom = 59.8e-3;

% Sets the calibration value for SYAG in eV/m
SYAG_cal = 64.4e9;
%% Loads dataset
dataloc = ['./',experiment,'/',experiment,'_',runname,'/',experiment,'_',runname,'.mat'];
load(dataloc);


% Extracts number of steps
stepsAll = data_struct.params.stepsAll;
if isempty(stepsAll);stepsAll = 1;end

% calculate xt calibration factor
xtcalibrationfactor = data_struct.metadata.DTOTR2.RESOLUTION*1e-6/streakFromGUI/3e8;
%% Grab XTCAV images on DTOTR2

% Combines the image data into one object
DTOTR2data = [];  % Initialize empty
baseFolder = './E338/';  % Update this path if needed

for a = 1:length(stepsAll)
    raw_path = data_struct.images.DTOTR2.loc{a};  % Cell indexing
    expr = [experiment '_\d+/images/DTOTR2/DTOTR2_data_step\d+\.h5'];
    
    % Extract relative path from full path string using regex
    match = regexp(raw_path, expr, 'match');
    
    if isempty(match)
        warning('Path did not match expected format: %s', raw_path);
        continue;
    end
    
    % Construct the full path to the HDF5 file
    DTOTR2datalocation = fullfile(baseFolder, match{1});
    
    % Read data from the HDF5 file
    stepData = h5read(DTOTR2datalocation, '/entry/data/data');
    
    % Convert to double precision
    stepData = double(stepData);
    
    % Concatenate along 3rd dimension
    if isempty(DTOTR2data)
        DTOTR2data = stepData;
    else
        DTOTR2data = cat(3, DTOTR2data, stepData);
    end
end

% Keeps only the data with a common index
DTOTR2commonind = data_struct.images.DTOTR2.common_index;
DTOTR2data = DTOTR2data(:,:,DTOTR2commonind);
nshots = size(DTOTR2data,3);

%% Rotate the DTOTR2 images so they look like they do on profmon GUIs
%xtcavImages = rotate_3d_array(DTOTR2data); % Nope no need to do this they are the right side up
xtcavImages = DTOTR2data-double(data_struct.backgrounds.DTOTR2);

%% Pre-crop the images by hand - they are too large and the autocrop fails
xmin = 200;
xmax = 800;
ymin = 200;
ymax = 800;
xtcavImages = xtcavImages(ymin:ymax,xmin:xmax,:);
%% Get the XTCAV phase on each shot (important if toggler is on)
[bsaScalarData,bsaVars] = extractDAQBSAScalars(data_struct);
ispv = regexp('TCAV_LI20_2400_A',bsaVars);
pvidx = find(~cellfun(@isempty,ispv));
xtcavAmpl = bsaScalarData(pvidx,:);

ispv = regexp('TCAV_LI20_2400_P',bsaVars);
pvidx = find(~cellfun(@isempty,ispv));
xtcavPhase = bsaScalarData(pvidx,:);

xtcavOffShots = xtcavAmpl<0.1;
xtcavPhase(xtcavOffShots) = 0; %Set this for ease of plotting

figure;
    yyaxis left
    plot(xtcavAmpl)
    ylabel('XTCAV Ampl [MV]')
    yyaxis right
    plot(xtcavPhase)
    ylabel('XTCAV Phase [deg]')
%% Plot sample images of the beam with tcav at +90 deg, 0 and -90 deg
% Find the first shot where tcav is at -90, 0 and +90 deg
near_minus_90_idx = find(xtcavPhase >= -91 & xtcavPhase <= -89, 2);  % First index near -90
near_plus_90_idx = find(xtcavPhase >= 89 & xtcavPhase <= 91, 2);     % First index near +90
zero_idx = find(xtcavPhase == 0, 1);                                 % First index for 0

sampleImageIndices = [near_minus_90_idx(2),zero_idx,near_plus_90_idx(2)];
plotTitles = {'Tcav phase -90 deg', ' 0 deg',' +90 deg'};

sampleImage =  squeeze(xtcavImages(:,:,sampleImageIndices(1)));

% Add some pre-processing step to the tcav data
hotPixThreshold = 1e3;
sigma = 1;
threshold = 5;

% Define the x and yrange for cropping the image; Need to automate this
% figure;imagesc(sampleImage)

xrange = 150;
yrange = xrange;

figure('Position', [100, 100, 1000, 500])
for ij = 1:3

   sampleImage =  squeeze(xtcavImages(:,:,sampleImageIndices(ij)));


   sampleImageProcessed = processNoisyTCAVImage(sampleImage,hotPixThreshold,sigma,threshold);

   sampleImageCropped = cropProfmonImg(sampleImageProcessed,xrange,yrange,0);

   horzProj = sum(sampleImageCropped,1);

   subplot(2,3,ij)
    imagesc(sampleImageCropped);colormap jetvar;
    title(plotTitles{ij})
   subplot(2,3,ij+3)
   plot(horzProj)

end
sgtitle(sprintf(['TCAV images before centroid correction DAQ ',experiment,' - ',runname]));
%% Calculate centroid correction
sampleImageTcavOff = squeeze(xtcavImages(:,:,sampleImageIndices(2)));
sampleImageTcavOff = cropProfmonImg(sampleImageTcavOff,xrange,yrange,0);

img = sampleImageTcavOff;
img = medfilt2(img);
sigma = 5;
filter_size = 2*ceil(3*sigma)+1;
gaussian_filter = fspecial('gaussian',filter_size,sigma);
processedImage = imfilter(img,gaussian_filter,'conv');
figure;subplot(2,1,1);imagesc(processedImage);subplot(2,1,2);imagesc(sampleImageTcavOff)

%Nrows = 2*xrange; % calculate centroid correction for every row
Nrows = size(sampleImageTcavOff,1);
[centroidIndices, centersOfMass] = segment_centroids_and_com(processedImage, Nrows,1);
%% Apply centroid correction to images and replot
figure('Position', [100, 100, 1000, 500])

for ij = 1:3

   sampleImage =  squeeze(xtcavImages(:,:,sampleImageIndices(ij)));

   sampleImageCropped = cropProfmonImg(sampleImage,xrange,yrange,0);

   centroidCorrections = round(centersOfMass-size(centersOfMass,1)/2);

   centroidCorrections(isnan(centroidCorrections)) = 0;

   sampleImageCroppedShifted = sampleImageCropped;

   sampleImageCroppedShifted = shiftRows(sampleImageCropped,-centroidCorrections);

   horzProj = sum(sampleImageCroppedShifted,1);

   subplot(2,3,ij)
    imagesc(sampleImageCroppedShifted);colormap jetvar;
    title(plotTitles{ij})
   subplot(2,3,ij+3)
   plot(horzProj)

end
sgtitle(sprintf(['TCAV images after centroid correction DAQ ',experiment,' - ',runname]));

%% Calculate the current profile normalizing the integral to the charge
isChargePV = regexp('TORO_LI20_2452_TMIT',bsaVars);
pvidx = find(~cellfun(@isempty,isChargePV));
charge = bsaScalarData(pvidx,:)*1.6e-19;  % in C

figure('Position', [100, 100, 1000, 500])

for ij = 1:3
    if ij==2;continue;end% Skip the tcav off case

    % Import image and apply centroid correction
   sampleImage =  squeeze(xtcavImages(:,:,sampleImageIndices(ij)));

   sampleImageProcessed = processNoisyTCAVImage(sampleImage,hotPixThreshold,sigma,threshold);

   sampleImageCropped = cropProfmonImg(sampleImageProcessed,xrange,yrange,1);

   centroidCorrections = round(centersOfMass-size(centersOfMass)/2);

   centroidCorrections(isnan(centroidCorrections)) = 0;

   %sampleImageCroppedShifted = sampleImageCropped;
   sampleImageCroppedShifted = shiftRows(sampleImageCropped,-centroidCorrections);

   % Calculate the current profile from the streaked projection

   streakedProfile = sum(sampleImageCroppedShifted,1);

   tvar = [1:length(streakedProfile)]*xtcalibrationfactor;
   tvar = tvar - median(tvar);% Center around 0
   prefactor = charge(sampleImageIndices(ij))/trapz(tvar,streakedProfile);

   currentProfile = streakedProfile*prefactor;

   subplot(2,3,ij)
       imagesc(tvar*1e15,1:size(sampleImageCroppedShifted,1),sampleImageCroppedShifted);colormap jetvar;
       title(plotTitles{ij})
       ylabel('y [pix]')
   subplot(2,3,ij+3)
       plot(tvar*1e15,currentProfile*1e-3)
       xlabel('Time [fs]')
       ylabel('Current [kA]')
end
sgtitle('TCAV images with current profile after centroid correction');

%% Loop thru all +90 and -90 deg shots and calculate the current profile
% Note here i'm assuming the centroid correction can be taken from a single
% tcav off image. A better treatment would find the nearest tcav off image
% in the dataset and use that to apply the centroid correction on each shot

minus_90_idx = find(xtcavPhase >= -91 & xtcavPhase <= -89);
plus_90_idx = find(xtcavPhase >= 89 & xtcavPhase <= 91);

for ij = 1:length(minus_90_idx)
   % Import image and apply centroid correction
   sampleImage =  squeeze(xtcavImages(:,:,minus_90_idx(ij)));

   sampleImageProcessed = processNoisyTCAVImage(sampleImage,hotPixThreshold,sigma,threshold);

   [sampleImageCropped,errorFlag] = cropProfmonImg(sampleImageProcessed,xrange,yrange,0);

   if errorFlag;currentProfile_minus_90(ij,:)= zeros(2*xrange,1);continue;end

   centroidCorrections = round(centersOfMass-size(centersOfMass,1)/2);

   centroidCorrections(isnan(centroidCorrections)) = 0;

   %sampleImageCroppedShifted = sampleImageCropped; % if u want to avoid centroid correction

   sampleImageCroppedShifted = shiftRows(sampleImageCropped,-centroidCorrections);

   % Calculate the current profile from the streaked projection

   streakedProfile = sum(sampleImageCroppedShifted,1);

   tvar = [1:length(streakedProfile)]*xtcalibrationfactor;
   tvar = tvar - median(tvar);% Center around 0
   prefactor = charge(minus_90_idx(ij))/trapz(tvar,streakedProfile);

   currentProfile_minus_90(ij,:) = 1e-3*streakedProfile*prefactor;% in kA
end

for ij = 1:length(plus_90_idx)
   % Import image and apply centroid correction
   sampleImage =  squeeze(xtcavImages(:,:,plus_90_idx(ij)));

   sampleImageProcessed = processNoisyTCAVImage(sampleImage,hotPixThreshold,sigma,threshold);

   [sampleImageCropped,errorFlag] = cropProfmonImg(sampleImageProcessed,xrange,yrange,0);


   if errorFlag;currentProfile_plus_90(ij,:)= zeros(2*xrange,1);continue;end

   centroidCorrections = round(centersOfMass-size(centersOfMass,1)/2);
   centroidCorrections(isnan(centroidCorrections)) = 0;

   sampleImageCroppedShifted = shiftRows(sampleImageCropped,-centroidCorrections);

   % Calculate the current profile from the streaked projection

   streakedProfile = sum(sampleImageCroppedShifted,1);

   tvar = [1:length(streakedProfile)]*xtcalibrationfactor;
   tvar = tvar - median(tvar);% Center around 0
   prefactor = charge(plus_90_idx(ij))/trapz(tvar,streakedProfile);

   currentProfile_plus_90(ij,:) = 1e-3*streakedProfile*prefactor;% in kA
end

figure;
    subplot(2,1,1)
    imagesc(tvar*3e8*1e6,minus_90_idx,currentProfile_minus_90);colormap jetvar;
    xlabel('z [um]')
    ylabel('Shot Number')
    title(plotTitles{1})
    h = colorbar;
    ylabel(h,'I [kA]')
    subplot(2,1,2)
    imagesc(tvar*3e8*1e6,plus_90_idx,currentProfile_plus_90);colormap jetvar;
    xlabel('z [um]')
    ylabel('Shot Number')
    title(plotTitles{3})
    h = colorbar;
    ylabel(h,'I [kA]')   
    sgtitle(sprintf(['TCAV current DAQ ',experiment,' - ',runname]))
%% Calculate the drive-witness separation and compare with BLEN
ispv = regexp('BLEN_LI14_888_BRAW',bsaVars);
pvidx = find(~cellfun(@isempty,ispv));
bc14BLEN = bsaScalarData(pvidx,:);
separationCutoff = 0.05;% fraction of the drive/witness peak current for filtering

[sorted_bc14BLEN, sort_BC14BLEN_indices] = sort(bc14BLEN);

for ij = 1:length(minus_90_idx)
    % Finds the distance between the peaks
    [~,locs,~,p] = findpeaks(currentProfile_minus_90(ij,:));
    [~,ind_max_2] = maxk(p,2);
    pos = locs(ind_max_2);

    if isempty(pos);bunchSeparation_minus_90(ij) = 0;continue;end

    peak_separation = abs(pos(1) - pos(2));

    if abs(p(ind_max_2(1))) * separationCutoff > abs(p(ind_max_2(2)))
        peak_separation = 0;
    end

    % Converts the separation from pixels to meters and saves the information
    bunchSeparation_minus_90(ij) = peak_separation*xtcalibrationfactor;
    currentRatio_minus_90(ij) = currentProfile_minus_90(ij,pos(1))./currentProfile_minus_90(ij,pos(2));

end

for ij = 1:length(plus_90_idx)
    % Finds the distance between the peaks
    [~,locs,~,p] = findpeaks(currentProfile_plus_90(ij,:));
    [~,ind_max_2] = maxk(p,2);

    pos = locs(ind_max_2);

    if isempty(pos);bunchSeparation_plus_90(ij) = 0;continue;end

    peak_separation = abs(pos(1) - pos(2));

    if abs(p(ind_max_2(1))) * separationCutoff > abs(p(ind_max_2(2)))
        peak_separation = 0;
    end

    % Converts the separation from pixels to meters and saves the information
    bunchSeparation_plus_90(ij) = peak_separation*xtcalibrationfactor;
    currentRatio_plus_90(ij) = currentProfile_plus_90(ij,pos(1))./currentProfile_plus_90(ij,pos(2));
end
%% Plot the bunch seaparation vs BLEN at each zero crossing
figure
    yyaxis left

    h1 = plot(plus_90_idx,bunchSeparation_plus_90*3e8*1e6,'*r');
    hold on
    h2 = plot(minus_90_idx,bunchSeparation_minus_90*3e8*1e6,'*');
    ylabel('Bunch Separation [um]')
    yyaxis right

    plot(bc14BLEN,'.')
    ylabel('BC14 BLEN')
    legend([h1,h2],{'+90 deg','-90 deg'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['TCAV bunch separation DAQ ',experiment,' - ',runname]))       
    xlim([1,length(bc14BLEN)])
    set(gcf,'color','white')
figure
    scatter(bc14BLEN(plus_90_idx),bunchSeparation_plus_90*3e8*1e6,'*r')
    hold on
    scatter(bc14BLEN(minus_90_idx),bunchSeparation_minus_90*3e8*1e6,'*')
    grid on
    xlabel('BC14 BLEN')
    ylabel('Bunch Separation [um]')
    legend('+90 deg','-90 deg','location','NorthWest')
    box on
    title(sprintf(['TCAV bunch separation DAQ ',experiment,' - ',runname]))     

figure

    h1 = plot(plus_90_idx,currentRatio_plus_90,'*r');
    hold on
h2 = plot(minus_90_idx,currentRatio_minus_90,'*b');
    ylabel('Peak Current Ratio')   
    legend([h1,h2],{'+90 deg','-90 deg'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['TCAV peak Current Ratio DAQ ',experiment,' - ',runname]))       
    set(gcf,'color','white')
    xlim([1,length(bc14BLEN)])
%% make the same plot as above but with the step value instead of blen

steps = data_struct.scalars.steps(data_struct.scalars.common_index);
scan_vals = data_struct.params.scanVals{1};
step_vals = scan_vals(steps);

figure
    yyaxis left

    h1 = plot(plus_90_idx,bunchSeparation_plus_90*3e8*1e6,'*r');
    hold on
    h2 = plot(minus_90_idx,bunchSeparation_minus_90*3e8*1e6,'*');
    ylabel('Bunch Separation [um]')
    yyaxis right

    plot(step_vals)
    ylabel('Step Value')
    legend([h1,h2],{'+90 deg','-90 deg'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['TCAV bunch separation DAQ ',experiment,' - ',runname]))       
    xlim([1,length(step_vals)])
    set(gcf,'color','white')
%% Calculate the charge in the drive and witness beams based on DTOTR projection

energyProjection = squeeze(sum(xtcavImages,2));

figure;
imagesc(energyProjection);colormap jetvar

idxDrive = 295; % index in the energy projection that separates Drive-witness - determine it by looking at the above waterfall plot

driveBeamEnergyProj = energyProjection(1:idxDrive,:);
witnessBeamEnergyProj = energyProjection(idxDrive:end,:);

chargeInDrive = charge.*sum(driveBeamEnergyProj,1)./sum(energyProjection,1);
chargeInWitness = charge.*sum(witnessBeamEnergyProj,1)./sum(energyProjection,1);

figure;
yyaxis left
h1 = plot(chargeInDrive*1e9,'.');
hold on
h2 = plot(chargeInWitness*1e9,'.r');
ylabel('Bunch Charge [nC]')
ylim([0,1])
% yyaxis right
% plot(bc14BLEN)
% ylabel('BC14 BLEN')
% xlabel('Shot Number')
% legend([h1,h2],{'Drive','Witness'},'location','NorthWest')
% title(sprintf(['DTOTR2 Bunch Charge DAQ ',experiment,' - ',runname]))

    yyaxis right

    plot(step_vals)
    ylabel('Step Value')
    legend([h1,h2],{'Drive','Witness'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['DTOTR2 Bunch Charge DAQ ',experiment,' - ',runname]))     
    xlim([1,length(step_vals)])
    set(gcf,'color','white')

%% Calculate the charge in the drive and witness beams based on SYAG projection
% Combines the SYAG data into one object
for a = 1:length(stepsAll)
    SYAGdatalocation = data_struct.images.SYAG.loc{a};
    if a == 1
        SYAGdata = double(h5read(SYAGdatalocation,'/entry/data/data'));
    else
        SYAGdatastep = double(h5read(SYAGdatalocation,'/entry/data/data'));
        SYAGdata(:,:,end+1:end+size(SYAGdatastep,3)) = SYAGdatastep;
    end
end
% Keeps only the data with a common index
SYAGcommonind = data_struct.images.SYAG.common_index;
SYAGdata = SYAGdata(:,:,SYAGcommonind);

syagImages = SYAGdata-double(data_struct.backgrounds.SYAG);
%syagImages = rotate_3d_array(syagImages); % Don't yet understand the
%orientation of SYAG in this dataset

syagProjection = squeeze(sum(syagImages,2));
%%
figure;
imagesc(syagProjection);

idxDrive = 670; % index in the energy projection that separates Drive-witness - determine it by looking at the above waterfall plot

driveBeamsyagProj = syagProjection(1:idxDrive,:);
witnessBeamsyagProj = syagProjection(idxDrive:end,:);

chargeInWitnessFromSyag = charge.*sum(witnessBeamsyagProj,1)./sum(syagProjection,1);
chargeInDriveFromSyag = charge.*sum(driveBeamsyagProj,1)./sum(syagProjection,1);

figure;
yyaxis left
h1 = plot(chargeInDriveFromSyag*1e9,'.');
hold on
h2 = plot(chargeInWitnessFromSyag*1e9,'.r');
ylabel('Bunch Charge [nC]')
ylim([0,1])

% yyaxis right
% plot(bc14BLEN)
% ylabel('BC14 BLEN')
% xlabel('Shot Number')
% legend([h1,h2],{'Drive','Witness'},'location','NorthWest')
% title(sprintf(['SYAG Bunch Charge DAQ ',experiment,' - ',runname]))

    yyaxis right

    plot(step_vals)
    ylabel('Step Value')
    legend([h1,h2],{'Drive','Witness'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['SYAG Bunch Charge DAQ ',experiment,' - ',runname]))     
    xlim([1,length(step_vals)])
    set(gcf,'color','white')
%% Compare the charge measurement on SYAG and DTOTR2

% Convert charges to nC
chargeInDrive_nC = chargeInDrive * 1e9;
chargeInDriveFromSyag_nC = chargeInDriveFromSyag * 1e9;
chargeInWitness_nC = chargeInWitness * 1e9;
chargeInWitnessFromSyag_nC = chargeInWitnessFromSyag * 1e9;

% Remove outliers from the first dataset
outliers1 = isoutlier(chargeInDrive_nC) | isoutlier(chargeInDriveFromSyag_nC);
chargeInDrive_nC_clean = chargeInDrive_nC(~outliers1);
chargeInDriveFromSyag_nC_clean = chargeInDriveFromSyag_nC(~outliers1);

% Remove outliers from the second dataset
outliers2 = isoutlier(chargeInWitness_nC) | isoutlier(chargeInWitnessFromSyag_nC);
chargeInWitness_nC_clean = chargeInWitness_nC(~outliers2);
chargeInWitnessFromSyag_nC_clean = chargeInWitnessFromSyag_nC(~outliers2);

% Fit a line to the first scatter plot
p1 = polyfit(chargeInDrive_nC_clean, chargeInDriveFromSyag_nC_clean, 1);
fitLine1 = polyval(p1, chargeInDrive_nC_clean);

% Fit a line to the second scatter plot
p2 = polyfit(chargeInWitness_nC_clean, chargeInWitnessFromSyag_nC_clean, 1);
fitLine2 = polyval(p2, chargeInWitness_nC_clean);

% Create the first subplot
figure('Position', [100, 100, 1000, 500])

subplot(1,2,1)
scatter(chargeInDrive_nC_clean, chargeInDriveFromSyag_nC_clean)
%hold on;
%plot(chargeInDrive_nC_clean, fitLine1, 'k-')
title(['Drive Charge'])
xlabel('Charge in Drive from DTOTR2 [nC]')
ylabel('Charge in Drive from Syag [nC]')
hold off;
grid on
% Create the second subplot
subplot(1,2,2)
scatter(chargeInWitness_nC_clean, chargeInWitnessFromSyag_nC_clean,'or')
%hold on;
%plot(chargeInWitness_nC_clean, fitLine2, 'k-')
title(['Witness Charge'])
xlabel('Charge in Witness from DTOTR2 [nC]')
ylabel('Charge in Witness from Syag [nC]')
grid on

hold off;
sgtitle(sprintf(['SYAG and DTOTR2 Bunch Charge DAQ ',experiment,' - ',runname]))
set(gcf,'color','white')


%% Calculate the energy separation on SYAG and plot it vs the bunch separation on DTOTR2

driveBeamsyagProj = syagProjection(1:idxDrive,:);
witnessBeamsyagProj = syagProjection(idxDrive:end,:);
xdrive = 1:idxDrive;
xwitness =  1:size(witnessBeamsyagProj,1);
for ij = 1:size(driveBeamsyagProj,2)
massHProjection = sum(driveBeamsyagProj(:,ij)); % Total mass of the projection
drive_centroid(ij) = sum(xdrive' .* driveBeamsyagProj(:,ij)) / massHProjection;


massHProjection = sum(witnessBeamsyagProj(:,ij)); % Total mass of the projection
%wit_com(ij) = sum(xwitness' .* witnessBeamsyagProj(:,ij)) / massHProjection;
%instead of the witness com just find the peak of the witness beam
[maxi,idxWit] = max(smooth(witnessBeamsyagProj(:,ij)));

wit_centroid(ij) = idxDrive+idxWit;

end

figure;
imagesc(syagProjection);hold on;
plot(drive_centroid);hold on;plot(wit_centroid)

energy_separation = (wit_centroid - drive_centroid)* data_struct.metadata.SYAG.RESOLUTION*1e-6* SYAG_cal; %in eV?
%%
figure;plot(energy_separation*1e-6 );ylabel('Drive Wit energy Separation [MeV]')
xlabel('Shot number');grid on;
sgtitle(sprintf(['SYAG Energy Separation DAQ ',experiment,' - ',runname]))
set(gcf,'color','white')
    xlim([1,length(energy_separation)])

    figure
    yyaxis left

    h1 = plot(plus_90_idx,bunchSeparation_plus_90*3e8*1e6,'*r');
    hold on
    h2 = plot(minus_90_idx,bunchSeparation_minus_90*3e8*1e6,'*');
    ylabel('Bunch Separation [um]')
    yyaxis right

    plot(energy_separation*1e-6,'--')
    ylim([110,275])
    ylabel('Energy Separation on SYAG [MeV]')
    legend([h1,h2],{'+90 deg','-90 deg'},'location','NorthWest')
    xlabel('Shot number')
    title(sprintf(['DTOTR2 bunch separation vs SYAG energy Separation DAQ ',experiment,' - ',runname]))       
    xlim([1,length(energy_separation*1e-6 )])
    set(gcf,'color','white')
%% Make a 2D heatmap of bunch separation vs step value and BLEN
[nonbsaScalarData,nonbsaVars] = extractDAQnonBSAScalars(data_struct);
% Grab the collimator position at each step
ispv = regexp('COLL_LI20_2069_MOTR_RBV',nonbsaVars);
pvidx = find(~cellfun(@isempty,ispv));
notchCollPosition = nonbsaScalarData(pvidx,:);

% scatter plot of the bunch separations
figure('Position', [100, 100, 1000, 500])
%imagesc(bc14BLEN(plus_90_idx),notchCollPosition(plus_90_idx),bunchSeparation_plus_90*3e8*1e6)
subplot(1,2,1)
scatter(step_vals(plus_90_idx),bunchSeparation_plus_90*3e8*1e6);ylim([25,250])
grid on
legend('-90 deg','location','northwest')

ylabel('Bunch Separation [um]')

subplot(1,2,2)
scatter(step_vals(minus_90_idx),bunchSeparation_minus_90*3e8*1e6,'r');ylim([25,250])
grid on   
title(sprintf(['DTOTR2 bunch separation vs shot number ',experiment,' - ',runname]))       
xlabel('Step Value')
ylabel('Bunch Separation [um]')
legend('+90 deg','location','northwest')
    set(gcf,'color','white')

%% 2d heatmap

% Define constants
N = 200;
caxis_limits = [0, 250];
figure_position = [100, 100, 1500, 500];
% Create figure
figure('Position', figure_position);

% Plot -90 deg data
plot2DbunchseparationVsCollimatorAndBLEN(bc14BLEN, step_vals, bunchSeparation_minus_90, minus_90_idx, 1, '- 90 deg',N,caxis_limits);

% Plot +90 deg data
plot2DbunchseparationVsCollimatorAndBLEN(bc14BLEN, step_vals, bunchSeparation_plus_90, plus_90_idx, 2, '+ 90 deg',N,caxis_limits);

% Add super title
sgtitle(sprintf('DTOTR2 bunch separation vs notch pos & BLEN %s - %s', experiment, runname));

%% Now fit a 2D function to the data an plot the surface

% Define the data
X = bc14BLEN(plus_90_idx);
Y = step_vals(plus_90_idx);
Z = bunchSeparation_plus_90 * 3e8 * 1e6;

% Filter good shots
goodShots = Z < 250 & Z > 0;
X = X(goodShots)';
Y = Y(goodShots)';
Z = Z(goodShots)';

% Combine X and Y into a matrix with two columns
XY = [X(:), Y(:)];

% Fit a polynomial surface to the data
fit_type = 'poly22'; % Second-degree polynomial
fit_model = fit(XY, Z, fit_type);

% Display the fit model
disp(fit_model);

% Create a grid for plotting
N = 200;
x = linspace(min(X), max(X), N);
y = linspace(min(Y), max(Y), N);
[Xi, Yi] = meshgrid(x, y);

% Evaluate the fitted model on the grid
Zi_fit = fit_model(Xi, Yi);

% Interpolate the original data on the grid
Zi_data = griddata(X, Y, Z, Xi, Yi);

% Create figure
figure('Position', [100, 100, 1500, 500]);

% Plot the original data
subplot(1, 2, 1);
surface(Xi, Yi, Zi_data, 'edgecolor', 'none');
colormap jet;
     cb = colorbar;
     cb.Label.String = 'Bunch Sep. [um]';
title('Original Data (+90 deg)');
xlabel('BC14 BLEN');
ylabel('Step Number (i.e. Coll. Position)');
zlabel('Bunch Separation [um]');
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);
caxis([0, 250]);

% Plot the fitted model
subplot(1, 2, 2);
surface(Xi, Yi, Zi_fit, 'edgecolor', 'none');
colormap jet;
     cb = colorbar;
     cb.Label.String = 'Bunch Sep. [um]';
title('Fitted Model (+90 deg)');
xlabel('BC14 BLEN');
ylabel('Step Number (i.e. Coll. Position)');
zlabel('Bunch Separation [um]');
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);
caxis([0, 250]);

% Add super title
sgtitle(sprintf('DTOTR2 bunch separation vs notch pos & BLEN %s - %s', experiment, runname));

% Save the fit model if you want
save('fit_model_separationVsCollPositionAndBLEN_DAQ_12710.mat', 'fit_model');


%% Now fit the bunch separation as a funcitno of energy separation   
xvals = linspace(150,250);
c = -190;
m = 1.55;
yvals = m*xvals+c;

    figure   
    plot(energy_separation(plus_90_idx)*1e-6,bunchSeparation_plus_90*3e8*1e6,'.')
    hold on
        plot(energy_separation(minus_90_idx)*1e-6,bunchSeparation_minus_90*3e8*1e6,'r.')
        ylabel('Bunch Separation [um]')
        xlabel('Energy Separation [MeV]')

        plot(xvals,yvals,'--k','LineWidth',4)
        grid on
        title(sprintf('Bunch separation [um] = 1.4* Energy Sep [MeV] -170'))
            set(gcf,'color','white')

%% Calculate the mean and std of the bunch separation for each step value and plot vs the mean and std of the SYAG energy separaiton



%% Pick a few shots for Kelly
% shotNums_minus90 = [417,717,906,1158];
% shotNums_plus90 = [392,689,883,1178];
% kellyshots = 125+250*[1:1:4];
%
% figure('Position', [100, 300, 1400, 900])
%
% for ij = 1:length(kellyshots)
%     % Pick some random shots
% %     a = find(minus_90_idx>kellyshots(ij),1);
% %     b = find(plus_90_idx>kellyshots(ij),1);
%
%     a = find(minus_90_idx==shotNums_minus90(ij),1);
%     b = find(plus_90_idx==shotNums_plus90(ij),1);
%
%     shotNum_minus90 = a(randi(length(a),1));
%     shotNum_plus90 = b(randi(length(b),1));
%     
%     dz_minus_90 = bunchSeparation_minus_90(shotNum_minus90)*3e8*1e6;
%     dz_plus_90 = bunchSeparation_plus_90(shotNum_plus90)*3e8*1e6;
%     
%     
%    subplot(2,4,ij)
%    plot(tvar*3e8*1e6,currentProfile_minus_90(shotNum_minus90,:));
%    xlim([-200,200])
%    ylim([0,15])
%    ylabel('Current [kA]')
%   
%    legend(sprintf(['Separation [um] = ',num2str(dz_minus_90,'%.0f')]))
%   
%    title(['Shot n. ',num2str(minus_90_idx(shotNum_minus90))])
%   
%       subplot(2,4,ij+4)
%    plot(tvar*3e8*1e6,currentProfile_plus_90(shotNum_plus90,:),'r');
%    %xlim([min(tvar*3e8*1e6),max(tvar*3e8*1e6)])
%    ylim([0,15])
%       xlim([-200,200])
%       xlabel('z [um]')
%       ylabel('Current [kA]')
%       title(['Shot n. ',num2str(plus_90_idx(shotNum_plus90))])
%             
%     legend(sprintf(['Separation [um] = ',num2str(dz_plus_90,'%.0f')]))
%     
%     
%     % Save the current profile on some shots
%     zvar = tvar*3e8*1e6; % in um
%     currplus90(ij,:) = currentProfile_plus_90(shotNum_plus90,:);
%     currminus90(ij,:) = currentProfile_minus_90(shotNum_minus90,:);
%     
%     
% end
%
% % save('zvar','zvar')
% % save('currplus90','currplus90');
% % save('currminus90','currminus90');


%% Calculate the bunch length in the drive and witness beams by fitting a biGaussian

y = currentProfile_minus_90(1,:);

% % Define the bi-Gaussian function
biGaussian = @(params, x) params(1)*exp(-(x - params(2)).^2/(2*params(3)^2)) + ...
                           params(4)*exp(-(x - params(5)).^2/(2*params(6)^2));

for ij = 1:length(minus_90_idx)
    % fit  bigaussian to the data
    y = currentProfile_minus_90(ij,:) ;
    x = [1:length(y)];% defined in the above example
    % Fit the bi-Gaussian model to the data
    options = optimset('Display','off'); % no output during fitting

     % % Initial parameters: [A1, mean1, sigma1, A2, mean2, sigma2]
    initial_guess = [max(y), 150, 4, max(y)*0.1, 150+ij*0.15, 4]; % initial guess

    [param_fit, residual, exitflag] = lsqcurvefit(@(params, x) biGaussian(params, x), initial_guess, x, y, [], [], options);

    % Extract parameters
    A1 = param_fit(1);
    mean1(ij) = param_fit(2);
    sigma1(ij) = param_fit(3);
    A2 = param_fit(4);
    mean2(ij) = param_fit(5);
    sigma2(ij) = param_fit(6);

    % Calculate separation between peaks
    separation = abs(mean1 - mean2);

    % Calculate the R^2 of the fit
    y_fit = biGaussian(param_fit,x);
    SST = sum((y - mean(y)).^2);  % Total sum of squares
    SSR = sum((y - y_fit).^2);    % Sum of squares of residuals
    R_squared(ij) = 1 - (SSR / SST);  % R^2

%    dels(ij) = separation*EOS_cal*3e8;
end

zvar = tvar*3e8*1e6; % in um
% Plot the sigma of the drive and witness
% Plot the data and the fit
figure;
plot(x, y, 'b.', 'DisplayName', 'Data'); hold on;
fitted_curve = biGaussian(param_fit, x);
plot(x, fitted_curve, 'r-', 'DisplayName', 'Fitted Bi-Gaussian');
legend show;
xlabel('x');
ylabel('y');
title('Bi-Gaussian Fit');
grid on;
sigmadrive_minus90 = sigma1*abs(zvar(2)-zvar(1));
sigmawit_minus90 = sigma2*abs(zvar(2)-zvar(1));



%         subplot(1,2,2);plot(minus_90_idx,sigmadrive./sigmawit,'k');
%     grid on;
%     ylabel('RMS bunch length Ratio')
%     xlabel('Shot Number ')
%     xlim([1,length(bc14BLEN)])
% subplot(1,2,2);
% yyaxis left
% plot(mean1,'b.');hold on;plot(mean2,'r.');ylim([0,100])
% yyaxis right
% plot(R_squared,'k.')


y = currentProfile_plus_90(1,:);

% % Define the bi-Gaussian function
biGaussian = @(params, x) params(1)*exp(-(x - params(2)).^2/(2*params(3)^2)) + ...
                           params(4)*exp(-(x - params(5)).^2/(2*params(6)^2));

for ij = 1:length(plus_90_idx)
    % fit  bigaussian to the data
    y = currentProfile_plus_90(ij,:) ;
    x = [1:length(y)];% defined in the above example
    % Fit the bi-Gaussian model to the data
    options = optimset('Display','off'); % no output during fitting

     % % Initial parameters: [A1, mean1, sigma1, A2, mean2, sigma2]
    initial_guess = [max(y), 150, 4, max(y)*0.1, 150-ij*0.15, 4]; % initial guess

    [param_fit, residual, exitflag] = lsqcurvefit(@(params, x) biGaussian(params, x), initial_guess, x, y, [], [], options);

    % Extract parameters
    A1 = param_fit(1);
    mean1(ij) = param_fit(2);
    sigma1p90(ij) = param_fit(3);
    A2 = param_fit(4);
    mean2(ij) = param_fit(5);
    sigma2p90(ij) = param_fit(6);

    % Calculate separation between peaks
    separation = abs(mean1 - mean2);

    % Calculate the R^2 of the fit
    y_fit = biGaussian(param_fit,x);
    SST = sum((y - mean(y)).^2);  % Total sum of squares
    SSR = sum((y - y_fit).^2);    % Sum of squares of residuals
    R_squared(ij) = 1 - (SSR / SST);  % R^2

%    dels(ij) = separation*EOS_cal*3e8;
end

% Plot the sigma of the drive and witness
% Plot the data and the fit
figure;
plot(x, y, 'b.', 'DisplayName', 'Data'); hold on;
fitted_curve = biGaussian(param_fit, x);
plot(x, fitted_curve, 'r-', 'DisplayName', 'Fitted Bi-Gaussian');
legend show;
xlabel('x');
ylabel('y');
title('Bi-Gaussian Fit');
grid on;
sigmadrive_plus90 = sigma1p90*abs(zvar(2)-zvar(1));
sigmawit_plus90 = sigma2p90*abs(zvar(2)-zvar(1));





figure;
    subplot(1,2,1);plot(minus_90_idx,sigmadrive_minus90,'b.');
    hold on;plot(minus_90_idx,sigmawit_minus90,'r.');ylim([0,50]);
    grid on;
    ylabel('RMS bunch length [um]')
    legend('Drive','Witness')
    xlabel('Shot Number ')
    xlim([1,length(bc14BLEN)])
    title(sprintf(['-90 deg '])) 


        subplot(1,2,2);plot(plus_90_idx,sigmadrive_plus90,'b.');
    hold on;plot(plus_90_idx,sigmawit_plus90,'r.');ylim([0,50]);
    grid on;
    ylabel('RMS bunch length [um]')
    legend('Drive','Witness')
    xlabel('Shot Number ')
    xlim([1,length(bc14BLEN)])
        title(sprintf(['+90 deg '])) 
    sgtitle(sprintf(['Drive-wit sigmaz  DAQ ',experiment,' - ',runname]))

    %% Prepare data for ML model prediction of the driver current

     driverPeak = max(currentProfile_plus_90,[],2);
     predictor = bsaScalarData(:,plus_90_idx);

     % Plot correlation coefficient between image data and all bsa scalar PVs
%cut out all the shots with l2 phase of 38 deg;
figure
for n=1:size(predictor,1)
   x = predictor(n,:); 
    y = driverPeak; % Egain
   R = corrcoef(x,y);
   c(n) = R(1,2);         
end
absc = abs(c);
[~,idx] = sort(absc);
h1 = figure
   plot(absc(idx),'LineWidth',2);grid on;box on
   ylabel('Correlation Coefficient')
   xticks([1:size(bsaScalarData,1)]);xlim([1,length(absc)])   
   xticklabels(bsaVars(idx));xtickangle(45)
   ax = gca(h1);
   ax.FontSize = 20;
   ax.XAxis.FontSize = 10;
    title(sprintf(['Corr. between CHER xctr and scalar PVs DAQ ',experiment,' - ',runname]))
    set(gcf,'color','white')

% Plot only values where |c|> some number
% Cut on the |c|>some number
%idx = absc>0.3;
%abscCut = absc(idx);

% Cut on top N correlated PVs where N = targetCount
targetCount = 15; % Number of

[someNumber, idx] = findThresholdForIndices(absc, targetCount);

abscCut = absc(idx);

bsaLabels = bsaVars(idx);
[~,idx] = sort(abscCut);
h1 = figure('Position', [100, 100, 900, 600]);
   plot(abscCut(idx),'ko-','LineWidth',2);grid on;box on
   ylabel('Correlation Coefficient')
   xticks([1:size(bsaLabels,1)]);xlim([1,length(abscCut)])   
   xticklabels(bsaLabels(idx));xtickangle(45)
   ax = gca(h1);
   ax.FontSize = 14;
   ax.XAxis.FontSize = 11;
   set(gca, 'TickLabelInterpreter', 'none','LineWidth',2)
   str = ['Corr. between CHER Egain and BSA Scalar PVs DAQ ',experiment,' - ',runname];
   title(sprintf(str),'interpreter','none')
      set(gcf,'color','white')
% For all those PVs with good correlation do a scatter plot of the driver pkI XCTR vs that PV     


     % Nofigure('Position', [100, 100, 1000, 800]);
for ij = 1:length(bsaLabels(idx))
   isPV = regexp(bsaLabels(idx(ij)),bsaVars);
   pvid = find(~cellfun(@isempty,isPV));

   subplot(5,3,ij)
       scatter(predictor(pvid,:),y,'.')
       xlabel(bsaLabels(idx(ij)),'interpreter','none');ylabel('Driver Ipk [kA]')
       box on
       grid on
end
   sgtitle(sprintf(['Scatter plot for highly correlated scalar PVs with driver Ipk ',experiment,' - ',runname]))
   set(gcf,'color','white')
     % with the driver peak current

     % Select out only the important PVs and feed those to the mode
     % doesnt work for some reason
     %predictors_for_ML = [predictor(idx,:);driverPeak'];% This is how ml regression learner likes the input data in a vector where the last row or column is the response and the rest are teh predictors

     predictors_for_ML = [predictor;driverPeak'];% This is how ml regression learner likes the input data in a vector where the last row or column is the response and the rest are teh predictors

 %% Import the model and predict the driver current

 load trainedModel_driverCurrent
 ytrue = driverPeak;
 ypred = trainedModel_driverCurrent.predictFcn(predictor);
 u=sum((ytrue-ypred).^2);
v=sum((ytrue-mean(ytrue)).^2);
score = 1-u./v;
[ytrueSorted,ind]=sort(ytrue);

 figure('Position', [100, 100, 1500, 500]);
h(1)=subplot(1,2,1)
    scatter(ypred,ytrue,'*');xlabel('Predicted Driver I pk [kA]');ylabel('Measured Driver I pk [kA]');grid on;
    hold on;plot(linspace(min(ytrue),max(ytrue)),linspace(min(ytrue),max(ytrue)),'k--','LineWidth',2);
    %ylim([150,350])
h(2)=subplot(1,2,2)
    plot(ypred(ind),'*');hold on;alpha(0.1)
    plot(ytrueSorted,'LineWidth',2);hold on
    xlabel('Shot Number (sorted on I pk)');ylabel('Driver I pk [kA]');
    legend(sprintf(['Predicted Value, accuracy = ',num2str(100*score,'%.0f'),'pct']),'Measured','Location','NorthWest');grid on;
    set(h(1:2),'FontSize',18,'box','on','LineWidth',2)
    xlim([1,length(ypred)]);%ylim([150,350])   
   sgtitle(sprintf(['ML Predicted Driver Peak Current ',experiment,' - ',runname]))
    set(gcf,'color','white') 
