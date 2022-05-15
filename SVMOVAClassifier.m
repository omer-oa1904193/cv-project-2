classdef SVMOVAClassifier< handle
    %SVMCLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % dataset
        X
        Y
    end
    
    properties (Access = private)
       models
    end
    
    methods
        function obj = SVMOVAClassifier()
            obj.models=containers.Map('KeyType','int');
        end
        
        
        function train(obj,X,Y)
            C=unique(Y);

            for i=1:size(C)
                Y_class=repmat(Y,1);
                Y_class(Y==C(i)) = 1;
                Y_class(Y~=C(i)) = -1;
                keySet{i} = C(i);
                valueSet{i} =fitPosterior(fitcsvm(X,Y_class));
            end            
            obj.models = containers.Map(keySet, valueSet);
        end
        
        function y_hat = predict(obj, x)
            classes=keys(obj.models);
            for i=1:size(classes,2)
                binary_label= predict(obj.models(classes{i}), x);
                if binary_label == 1
                    y_hat =classes{i};
                    return;
                end
            end
            
            y_hat = 0;
            %y_hat = randsample(classes,1);
        end
    end
end

