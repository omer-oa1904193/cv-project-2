function script()
    dataset=Dataset(0.7);
    %classifier = KNNClassifier(3, false, 'euclidean');
    classifier = SVMOVAClassifier();
    classifier.train(dataset.X_train, dataset.Y_train);
    
    disp("Training:")
    for i=1:size(dataset.X_train,1)
        x=dataset.X_train(i,:);
        y=dataset.Y_train(i);
        y_hat=classifier.predict(x);
        fprintf("true %d predicted %d\n", y, y_hat);
    end
    
    disp("Testing:")
    for i=1:size(dataset.X_test,1)
        x=dataset.X_test(i,:);
        y=dataset.Y_test(i);
        y_hat=classifier.predict(x);
        fprintf("true %d predicted %d\n", y, y_hat);
    end
end