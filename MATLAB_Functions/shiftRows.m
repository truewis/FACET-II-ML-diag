function shiftedImg = shiftRows(img, shifts)
    % Shift each row of the image horizontally by a given amount
    % img: 2D image matrix
    % shifts: vector of shift values, one per row (positive = right, negative = left)

    [nRows, nCols] = size(img);
    shiftedImg = zeros(size(img));

    for i = 1:nRows
        shift = shifts(i);
        row = img(i, :);
        if shift > 0
            shiftedImg(i, shift+1:end) = row(1:end-shift);
        elseif shift < 0
            shiftedImg(i, 1:end+shift) = row(-shift+1:end);
        else
            shiftedImg(i, :) = row;
        end
    end
end
