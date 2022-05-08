function script()
global GREEN_HUE_MIN;
global GREEN_HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
GREEN_HUE_MIN=60;
GREEN_HUE_MAX=165;
SATURATION_MIN=8;
VALUE_MIN=10;
COLOR_TO_LUM_MIN_THRESH=130;
    feature_1_x=zeros(30,1);
    feature_2_y=zeros(30,1);
    feature_3_z=zeros(30,1);
    for i=1:30
        img=imread(sprintf('images/%02d.jpg', i));
        green_count=count_green(img);
        if green_count>=COLOR_TO_LUM_MIN_THRESH
            img_bw=color_segment_image(img);
            feature_1_x(i)=green_count;
        else
            img_bw=lum_segment_image(img);
        end
        % clean up image
        img_bw=bwareafilt(img_bw,1);
        img_bw=(~bwareafilt(~img_bw,1));
        
        imshow(img_bw)
        img_lebeled = bwlabel(img_bw);
        img_props = regionprops(img_lebeled, 'Perimeter','Circularity','Eccentricity');
        feature_2_y(i)=img_props(1).Circularity;
        feature_3_z(i)=img_props(1).Eccentricity;
        
        if green_count<COLOR_TO_LUM_MIN_THRESH
            feature_1_x(i)=sum(sum(img_bw));
        end
    end
    feature_1_x=feature_1_x/max(feature_1_x);
    feature_2_y=feature_2_y/max(feature_2_y);
    feature_3_z=feature_3_z/max(feature_3_z);
    
    close all;
    figure;
    hold on;
    scatter3(feature_1_x(1:10), feature_2_y(1:10),feature_3_z(1:10) ,'r.');
    scatter3(feature_1_x(11:20), feature_2_y(11:20),feature_3_z(11:20) ,'g.');
    scatter3(feature_1_x(21:30), feature_2_y(21:30),feature_3_z(21:30) ,'b.');
    
    xlabel('Feature 1');
    ylabel('Feature 2');
    zlabel('Feature 3');
    hold off;
    grid();
    writematrix(feature_1_x,'feature_1.csv');
    writematrix(feature_2_y,'feature_2.csv');
    writematrix(feature_3_z,'feature_3.csv');
end


function out=count_green(img)
global GREEN_HUE_MIN;
global GREEN_HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
img=rgb2hsv(img);
b = ((img(:,:,1)*360)>=GREEN_HUE_MIN & (img(:,:,1)*360)<=GREEN_HUE_MAX)&...
    (img(:,:,2)*100)>=SATURATION_MIN & (img(:,:,3)*100)>= VALUE_MIN;
out=sum(sum(sum(b)));
end

function out=color_segment_image(img)
global GREEN_HUE_MIN;
global GREEN_HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
img=rgb2hsv(img);
mask = ((img(:,:,1)*360)>=GREEN_HUE_MIN & (img(:,:,1)*360)<=GREEN_HUE_MAX)&...
    (img(:,:,2)*100)>=SATURATION_MIN & (img(:,:,3)*100)>= VALUE_MIN;
%out=hsv2rgb(img) .* mask;
out=~mask;
end

function out=lum_segment_image(img)
img_gray=rgb2gray(img);
mask=uint8(~imbinarize(img_gray));
%out=img.*mask;
out=~mask;
end



function generate_greens()
x=100;
y=100;
image=zeros(x, y, 3);
for i=1:x
    for j=1:y
        r=randsample(0:255,1)/255;
        g=randsample(0:255,1)/255;
        b=randsample(0:255,1)/255;
        while ~(is_green([r g b]))
            r=randsample(0:255,1)/255;
            g=randsample(0:255,1)/255;
            b=randsample(0:255,1)/255;
        end
        image(i,j,1)=r;
        image(i,j,2)=g;
        image(i,j,3)=b;
    end
end
image=image/255;
imshow(image);
%disp(is_green(hsv2rgb([0.6 1 1])));
end

function out = is_green(hsv_pixel)
global GREEN_HUE_MIN;
global GREEN_HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
hue=hsv_pixel(1)*360;
saturation=hsv_pixel(2)*100;
value=hsv_pixel(3)*100;
out = (hue >= GREEN_HUE_MIN && hue <= GREEN_HUE_MAX && saturation>=SATURATION_MIN && value >= VALUE_MIN);
end
