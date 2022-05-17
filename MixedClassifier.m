classdef MixedClassifier < Classifier
    properties
        % dataset
        X
        Y
        name="Mixed"
        algorithm="Mixed(KNN/LSE/SVM) with equally weighted voting"
    end
    properties (Access=private)
        knn_classifier
        lse_classifier
        svm_classifier
    end
    
    methods
        function obj = MixedClassifier(KNN_K, KNN_distance_metric)
            obj.knn_classifier = KNNClassifier(KNN_K, false, KNN_distance_metric);
            obj.lse_classifier = LSEOVAClassifier();
            obj.svm_classifier = SVMOVAClassifier();
        end
        
        function train(obj,X,Y)
            obj.knn_classifier.train(X, Y);
            obj.lse_classifier.train(X, Y);
            obj.svm_classifier.train(X, Y);
        end
        
        function y_hat = predict(obj, x)
            knn_prediction=obj.knn_classifier.predict(x);
            lse_prediction=obj.lse_classifier.predict(x);
            svm_prediction=obj.svm_classifier.predict(x);
            y_hat=mode([knn_prediction lse_prediction svm_prediction]);
        end
    end
end

