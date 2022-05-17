classdef KNNClassifier < Classifier
    properties
        % dataset
        X
        Y
        % hyper paramaters
        K
        use_kmeans
        distance_metric
        name="KNN"
        algorithm="KNN"
    end
    
    methods
        function obj = KNNClassifier(K, use_kmeans, distance_metric)
            obj.K = K;
            obj.use_kmeans = use_kmeans;
            obj.distance_metric=distance_metric;
        end
        
        function train(obj,X,Y)
            if obj.use_kmeans
                num_of_classes=max(Y);
                dm=obj.distance_metric;
                if obj.distance_metric=="euclidean"
                    dm="sqeuclidean";
                end
                % prevent inifinte loop :(
                max_iter=100;
                iter_count=1;
                % equivelent of do-while
                while 1
                    % reset random stream
                    [~, centroids] =  kmeans(X, num_of_classes,'Distance',dm);
                    centroid_classes=repmat(-1,num_of_classes ,1);
                    % temporarily override use_kmeans, and X just here to
                    % use predict method
                    obj.use_kmeans=false;
                    obj.X=X;
                    obj.Y=Y;
                    for i=1:num_of_classes
                        centroid_classes(i)=obj.predict(centroids(i, :));
                    end
                    obj.use_kmeans=true;
                    
                    % keep repeating KMeans clustering until cluster centroids represent different classes
                    if sum(ismember([1:num_of_classes],centroid_classes))>=num_of_classes || iter_count>max_iter
                        break
                    end
                    iter_count=iter_count+1;
                end
                saved_X=centroids;
                saved_Y=centroid_classes;
            else
                saved_X=X;
                saved_Y=Y;
            end
            obj.X=saved_X;
            obj.Y=saved_Y;
        end
        
        function y_hat = predict(obj, x)
            if obj.use_kmeans
                k=1;
            else
                k=obj.K;
            end
            distances=[];
            for i=1:size(obj.X,1)
                distances=[distances; pdist([obj.X(i,:); x],obj.distance_metric)];
            end
            [~,sorting_idx]=sort(distances);
            sorted_neighbors=obj.Y(sorting_idx);
            nearest_neighbors=sorted_neighbors(1:k);
            y_hat=mode(nearest_neighbors);
        end
    end
end

