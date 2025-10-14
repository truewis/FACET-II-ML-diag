function processedImage = processNoisyTCAVImage(img,hotPixThreshold,sigma,threshold)
%PROCESSNOISYTCAVIMAGE Summary of this function goes here
%   Detailed explanation goes here

% Remove hot pixels
hotPixels = img>hotPixThreshold;
filteredImg = medfilt2(img);
correctedImage = filteredImg;
correctedImage(hotPixels) = filteredImg(hotPixels);
%figure;subplot(2,1,1);imagesc(img);subplot(2,1,2);imagesc(corrected_img)

% Apply Gaussian Blur with defined sigma
filter_size = 2*ceil(3*sigma)+1;
gaussian_filter = fspecial('gaussian',filter_size,sigma);
processedImage = imfilter(correctedImage,gaussian_filter,'conv');
processedImage = correctedImage;%Don't do the blur
% Apply thresholding
processedImage(processedImage<threshold) = 0.0;

% Plot result
%figure;subplot(2,1,1);imagesc(img);colormap jetvar;subplot(2,1,2);
%imagesc(processedImage);colormap jetvar
end
