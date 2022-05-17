classdef LSEOVAClassifier < handle
    properties
        % dataset
        X
        Y
        name="LSE"
        algorithm="LSE with One Versus All Reduciton"
    end
    
    properties (Access = private)
        models
    end
    
    methods
        function obj = LSEOVAClassifier()
            
        end
        
        
        function train(obj,X,Y)
            C=unique(Y);

            for i=1:size(C)
                Y_class=repmat(Y,1);
                Y_class(Y==C(i)) = 1;
                Y_class(Y~=C(i)) = -1;
                W = pinv(X) * Y_class;
                
                keySet{i} = C(i);
                valueSet{i} = W;
            end            
            obj.models = containers.Map(keySet, valueSet);
        end
        
        function y_hat = predict(obj, x)
            classes=keys(obj.models);
            highest_score=-Inf;
            closest_class=-Inf;
            for i=1:size(classes,2)
                binary_score=dot(obj.models(classes{i}), x);
                if binary_score > 0
                    y_hat =classes{i};
                    return;
                elseif binary_score > highest_score
                    highest_score=binary_score;
                    closest_class=classes{i};
                end
            end
            % if no class is picked choose the class that came closest
            y_hat = closest_class;
        end
    end
end

