function [centroidIndices, centersOfMass] = segment_centroids_and_com(img, Nrows, plotFlag)
    % Divide image into Nrows segments and compute the centroid of each row
    img = double(img);
    [nRows, nCols] = size(img);

    % For each row, compute weighted x-position (center of mass)
    centersOfMass = zeros(nRows, 1);
    for i = 1:nRows
        row = img(i, :);
        x = 1:nCols;
        if sum(row) == 0
            centersOfMass(i) = NaN;
        else
            centersOfMass(i) = sum(row .* x) / sum(row);
        end
    end

    centroidIndices = (1:nRows)';

    if plotFlag
        figure;
        plot(centroidIndices, centersOfMass);
        xlabel('Row Index');
        ylabel('Center of Mass');
        title('Row-wise Centroids (Center of Mass)');
    end
end
