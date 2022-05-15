classdef Dataset<handle
    
    properties
        X_all
        Y_all
        
        % not normalized
        X_original
        
        X_train
        Y_train
        X_test
        Y_test
    end
    
    properties (Constant)
        BACKUP_FILE="my_data.csv";
        
        % segmentation  constants
        GREEN_HUE_MIN=60;
        GREEN_HUE_MAX=165;
        SATURATION_MIN=8;
        VALUE_MIN=10;
        COLOR_TO_LUM_MIN_THRESH=130;
   end
    
    methods
        function obj = Dataset(training_percent)
            obj.load_images();
            obj.X_train=[];
            obj.Y_train=[];
            obj.X_test=[];
            obj.Y_test=[];
            for i=1:3
                label_examples = obj.X_all( ((i-1)*10)+1 : 10*i, :);
                [n_all, ~] = size(label_examples);
                n_train = n_all * training_percent;
                n_test = n_all - n_train;

                training_indexes=randperm(n_all,n_train);
                testing_indexes=setdiff([1:n_all], training_indexes);

                obj.X_train=[obj.X_train; label_examples(training_indexes, :)];
                obj.Y_train=[obj.Y_train; repmat(i, n_train,1)];

                obj.X_test=[obj.X_test; label_examples(testing_indexes, :)];
                obj.Y_test=[obj.Y_test; repmat(i, n_test,1)];
            end
        end
        
        function x=extract_features(obj, img, normalize)
            green_count=Dataset.count_green(img);
            if green_count>=Dataset.COLOR_TO_LUM_MIN_THRESH
                img_bw=Dataset.color_segment_image(img);
                f1=green_count;
            else
                img_bw=Dataset.lum_segment_image(img);
            end
            % clean up image

            img_bw=bwareafilt(img_bw,1);
            img_bw=(~bwareafilt(~img_bw,1));
            imshow(img_bw)

            img_lebeled = bwlabel(img_bw);

            img_props = regionprops(img_lebeled, 'Circularity','EulerNumber');
            f2=img_props(1).Circularity;
            f3=img_props(1).EulerNumber;

            if green_count<Dataset.COLOR_TO_LUM_MIN_THRESH
                f1=sum(sum(img_bw));
            end
            if normalize
                x=obj.normalize_x(f1,f2,f3);
            else
                x=[f1 f2 f3];
            end
        end
        
    end
    methods (Access = private)
        function load_images(obj)
            try
                data=readmatrix(Dataset.BACKUP_FILE);
                obj.X_original=data(:,1:size(data,2)-1);
                f1s=obj.X_original(:, 1);
                f2s=obj.X_original(:, 2);
                f3s=obj.X_original(:, 3);
                
                f1s=f1s/max(f1s);
                f2s=f2s/max(f2s);
                f3s=(f3s+abs(min(f3s)))/(max(f3s)+abs(min(f3s)));
                obj.X_all = [f1s f2s f3s];
                obj.Y_all=data(:,size(data,2));
                return
            catch
            end
            
            f1s=zeros(30,1);
            f2s=zeros(30,1);
            f3s=zeros(30,1);
            for i=1:30
                img=imread(sprintf('images/%02d.jpg', i));
                x=obj.extract_features(img, false);
                f1s(i)=x(1);
                f2s(i)=x(2);
                f3s(i)=x(3);
            end
            eval("@(f1,f2,f3) [f1/max(f1s) f2/max(f2s) (f3+abs(min(f3s)))/(max(f3s)+abs(min(f3s)))];")
            
            obj.X_original= [f1s f2s f3s];
            f1s=f1s/max(f1s);
            f2s=f2s/max(f2s);
            f3s=(f3s+abs(min(f3s)))/(max(f3s+abs(min(f3s))));
            Dataset.plot_features(f1s,f2s,f3s);
            grid();
            obj.X_all = [f1s f2s f3s];
            obj.Y_all=[repmat(1, 10,1); repmat(2, 10,1); repmat(3, 10,1);];
            data=[obj.X_original obj.Y_all];
            writematrix(data,Dataset.BACKUP_FILE);
        end

        function x = normalize_x(obj,f1, f2,f3)
            f1s=obj.X_original(:,1);
            f2s=obj.X_original(:,2);
            f3s=obj.X_original(:,3);
            x=[f1/max(f1s) f2/max(f2s) (f3+abs(min(f3s)))/(max(f3s)+abs(min(f3s)))];
        end
    end
    
    methods(Static)
        function out=count_green(img)
            img=rgb2hsv(img);
            b = ((img(:,:,1)*360)>=Dataset.GREEN_HUE_MIN & (img(:,:,1)*360)<=Dataset.GREEN_HUE_MAX)&...
                (img(:,:,2)*100)>=Dataset.SATURATION_MIN & (img(:,:,3)*100)>= Dataset.VALUE_MIN;
            out=sum(sum(sum(b)));
        end

        function out=color_segment_image(img)
            img=rgb2hsv(img);
            mask = ((img(:,:,1)*360)>=Dataset.GREEN_HUE_MIN & (img(:,:,1)*360)<=Dataset.GREEN_HUE_MAX)&...
                (img(:,:,2)*100)>=Dataset.SATURATION_MIN & (img(:,:,3)*100)>= Dataset.VALUE_MIN;
            %out=hsv2rgb(img) .* mask;
            out=~mask;
        end

        function out=lum_segment_image(img)
            img_gray=rgb2gray(img);
            mask=uint8(~imbinarize(img_gray));
            %out=img.*mask;
            out=~mask;
        end
        
        function plot_features(feature_1_x,feature_2_y,feature_3_z)
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
        end
    end
end

