function train(algorithm, hyper_params)
    if nargin < 3
        algorithm="KNN";
        hyper_params=struct("K",3,"use_kmeans",false);
    end
    data=readmatrix('training.csv');
    [n, d] = size(data);
    X=data(:,1:d-1);
    Y=data(:,d);
    
    if algorithm=="LSE"
        train_KNN(X,Y)
    elseif algorithm=="KNN"
        train_KNN(X,Y, hyper_params.K, hyper_params.use_kmeans)
    elseif algorithm=="SVM"
        train_SVM(X,Y)
    elseif algorithm=="PCA"
        train_PCA(X,Y)
    end
end


function train_LSE(X,Y)

end

function train_KNN(X,Y, K, use_kmeans)
    if use_kmeans
        num_of_classes=max(Y);
        % equivelent of do-while
        while 1 
            [indexes, centroids] =  kmeans(X, num_of_classes);
            centroid_classes=repmat(-1,num_of_classes ,1);
            for i=1:num_of_classes
                centroid_classes(i)=mode(Y(find(indexes==i),:));
            end
            % keep repeating KMeans clustering until cluster centroids represent different classes
            if sum(ismember([1:num_of_classes],centroid_classes))>=num_of_classes
                break
            end
        end
        saved_X=centroids;
        saved_Y=centroid_classes;
    else
        saved_X=X;
        saved_Y=Y;
    end
    
    writematrix([saved_X saved_Y], 'knn_model.csv')
    knn_meta=struct("K",K,"use_kmeans",use_kmeans);
    save("knn_meta.mat", '-struct', 'knn_meta')
end


function train_SVM(X,Y)

end

function train_PCA(X,Y)

end
