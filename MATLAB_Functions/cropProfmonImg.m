function  [croppedImg,errorFlag] = cropProfmonImg(img,xrange,yrange,plotFlag)
    img = double(img);
    mass = sum(sum(img));
    [x,y] = meshgrid(1:size(img,2),1:size(img,1));
    
        x_com = round(sum(sum(img.*x))/mass);
        y_com = round(sum(sum(img.*y))/mass);
    
    
        % Try finding COM by doing the max of the horz and ver Proj
        horzProj = sum(img,1); [~,x_com] = max(horzProj);
        vertProj = sum(img,2); [~,y_com] = max(vertProj);
    
    
        indx = [x_com:x_com+2*xrange-1]-xrange;
        indy = [y_com:y_com+2*yrange-1]-yrange;
    
    %     if indx(1)<1
    %         indx = indx+abs(indx(1))+1;
    %     end
    %     if indy(1)<1
    %         indy = indy+abs(indy(1))+1;
    %     end
    %
    %
    %     if indx(end)>size(img,1)
    %     indx = indx-abs(indx(end))+size(img,1);
    %     end
    %
    %     if indy(end)>size(img,2)
    %     indy = indy-abs(indy(end))+size(img,2);
    %     end
        try
        croppedImg = img(indy,indx);
        errorFlag = 0;
        catch
        warning('could not crop image, indices out of range')
            croppedImg = img;
            errorFlag = 1;
        end
    
    if plotFlag
        figure
        subplot(2,1,1)
        imagesc(img)
        subplot(2,1,2)
        imagesc(croppedImg)
    end
    
    end