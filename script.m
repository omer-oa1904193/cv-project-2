function script()
global HUE_MIN;
global HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
HUE_MIN=65;
HUE_MAX=165;
SATURATION_MIN=15;
VALUE_MIN=10;
    x=1:30;
    y=zeros(30);
    for i=1:30
        img=imread(sprintf('images/%02d.jpg', i));
        y(i)=count_green(img);
        %imshow(segment_image(img));
    end
    bar(x, y, 'BarWidth');
end


function out=count_green(img)
global HUE_MIN;
global HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
img=rgb2hsv(img);
b = ((img(:,:,1)*360)>=HUE_MIN & (img(:,:,1)*360)<=HUE_MAX & (img(:,:,2)*100)>=SATURATION_MIN & (img(:,:,3)*100)>= VALUE_MIN);
out=sum(sum(sum(b)));
end

function out=segment_image(img)
global HUE_MIN;
global HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
img=rgb2hsv(img);
mask =  ((img(:,:,1)*360)>=HUE_MIN & (img(:,:,1)*360)<=HUE_MAX & (img(:,:,2)*100)>=SATURATION_MIN & (img(:,:,3)*100)>= VALUE_MIN);
out=hsv2rgb(img) .* mask;
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
global HUE_MIN;
global HUE_MAX;
global SATURATION_MIN;
global VALUE_MIN;
hue=hsv_pixel(1)*360;
saturation=hsv_pixel(2)*100;
value=hsv_pixel(3)*100;
out = (hue >= HUE_MIN && hue <= HUE_MAX && saturation>=SATURATION_MIN && value >= VALUE_MIN);
end
