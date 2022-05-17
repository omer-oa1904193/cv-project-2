classdef SVMOVAClassifier< handle
    properties
        % dataset
        X
        Y
        name="SVM"
        algorithm="SVM with One Versus All Reduciton"
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
                warning('off',warning('query','last').identifier)
            end            
            obj.models = containers.Map(keySet, valueSet);
        end
        
        function y_hat = predict(obj, x)
            classes=keys(obj.models);
            highest_prob=-Inf;
            closest_label=-Inf;
            for i=1:size(classes,2)
                [binary_label, probs]= predict(obj.models(classes{i}), x);
                if binary_label == 1
                    y_hat =classes{i};
                    return;
                elseif binary_label == -1
                    if min(probs)>highest_prob
                        highest_prob=min(probs);
                        closest_label=classes{i};
                    end
                    
                end
            end
            
            y_hat = closest_label;
            %y_hat = randsample(classes,1);
        end
    end
end

