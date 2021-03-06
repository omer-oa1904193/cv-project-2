function script()
    dataset=Dataset(0.7);
    disp("CMPS 408 Project 2");
    disp("1- Train and Evaluate classifier");
    disp("2- Compare 2 classifiers");
    disp("Please select option: ");
    choice= input("");
    while choice<1 || choice>2
        disp("Please enter a valid value.");
        choice= input("");
    end
    if choice==1
        disp("Loading Data...")
        dataset=Dataset(0.7);
        disp("Please select a classifier: ");
        classifier=select_classifer();
        disp("Training...");
        classifier.train(dataset.X_train, dataset.Y_train);
        show_eval(dataset, classifier);
    elseif choice==2
        disp("Select clssifier 1:");
        classifier1 = select_classifer();
        fprintf("\n");
        disp("Select clssifier 2:");
        classifier2 = select_classifer();
        fprintf("\n");
        metric=input("What is the metric you want to compare? ('MacroF1'(default), 'OverallAccuracy'): ");
        if isempty(metric)
            metric='MacroF1';
        end
        iterations=input("How many iterations do you want to evaluate for? (default: 100)");
        if isempty(iterations)
            iterations=100;
        end
        disp("Training...");
        classifier1.train(dataset.X_train, dataset.Y_train);
        classifier2.train(dataset.X_train, dataset.Y_train);
        
        compare_classifiers(classifier1, classifier2, metric,iterations);
        
    end
    
end

function classifier=select_classifer()
    disp("1- K Nearest Neighbors (KNN)");
    disp("2- Support Vector Machines (SVM)");
    disp("3- Least Square Estimate (LSE)");
    disp("4- Mixed (KNN/SVM/LSE) with equal weighted voting");
    choice= input("");
    while choice<1 || choice>4
        disp("Please enter a valid value.");
        choice= input("");
    end
    
    if choice==1
        disp("Please enter hyperparamaters (enter nothing to use default values): ");
        K=input("K (default: 3): ");
        if isempty(K)
            K=3;
        end
        use_kmeans=input("Use K-Means 0/1? (default: false): ");
        if isempty(use_kmeans)
            use_kmeans=false;
        end
        distance_metric=input("Distance Metric (default: 'euclidean'): ");
        if isempty(distance_metric)
            distance_metric='euclidean';
        end
        classifier = KNNClassifier(K, use_kmeans, distance_metric);
    elseif choice==2
        classifier = SVMOVAClassifier();
    elseif choice==3
        classifier = LSEOVAClassifier();
    elseif choice==4
        disp("Please enter hyperparamaters (enter nothing to use default values): ");
        K=input("K (default: 3): ");
        if isempty(K)
            K=3;
        end
        distance_metric=input("Distance Metric (default: 'euclidean'): ");
        if isempty(distance_metric)
            distance_metric='euclidean';
        end
        classifier= MixedClassifier(K, distance_metric); 
    end
end

function compare_classifiers(classifier1, classifier2, metric, iterations)
    fprintf("Comparing classifiers %s vs %s with metric %s over %d iterations\n", classifier1.name, classifier2.name, metric,iterations);
    scores1=zeros(iterations,1);
    scores2=zeros(iterations,1);
    
    fprintf("Please wait...\n");
    % evaluation done many times to counter randomness of train/test split
    for i=1:iterations
        dataset=Dataset(0.7);
        scores1(i)=jsondecode(evaluate_classifier(classifier1, dataset.X_test, dataset.Y_test, false)).(metric);
        scores2(i)=jsondecode(evaluate_classifier(classifier2, dataset.X_test, dataset.Y_test, false)).(metric);
    end
    avg_score1=sum(scores1)/iterations;
    avg_score2=sum(scores2)/iterations;
    
    fprintf("Please wait...\n");
    fprintf("Average %s over %d iterations\n", metric, iterations);
    fprintf("%s: %f\n", classifier1.name,avg_score1);
    fprintf("%s: %f\n", classifier2.name,avg_score2);
    if avg_score1>avg_score2
        fprintf("%s wins.\n", classifier1.name);
    elseif avg_score2>avg_score1
        fprintf("%s wins.\n", classifier2.name);
    else
        fprintf("It's a tie.\n");
    end
end

function show_eval(dataset,classifier)
    fprintf("Evaluating %s classifier\n", classifier.name)
    disp("Training:")
    evaluate_classifier(classifier, dataset.X_train, dataset.Y_train);
    disp("--------------------");
    disp("Testing:")
    evaluate_classifier(classifier, dataset.X_test, dataset.Y_test);
end

function eval_json=evaluate_classifier(classifier, X_eval, Y_eval, verbose)
    if nargin==3
        verbose=true;
    end
    n_classes=size(unique(Y_eval),1);
    % rows: actual, columns: predicted
    confusion_matrix=zeros(n_classes, n_classes);
    for i=1:size(X_eval,1)
        x = X_eval(i,:);
        y = Y_eval(i);
        y_hat=classifier.predict(x);
        %fprintf("true %d predicted %d\n", y, y_hat);
        confusion_matrix(y, y_hat)=confusion_matrix(y, y_hat)+1;
    end
    if verbose
        disp("Confusion Matrix");
        disp(confusion_matrix);
    end
    total_f1=0;
    class_json_str=strings(1,n_classes);
    for i=1:n_classes
        TP=confusion_matrix(i,i);
        FP=sum(confusion_matrix(:,i)) - TP;
        FN=sum(confusion_matrix(i,:)) - TP;
        TN= sum(sum(confusion_matrix))-(TP+FP+FN);
        if verbose
            fprintf("Class %d\n",i);
            fprintf("True Positive: %d\n",TP);
            fprintf("True Negative: %d\n",TN);
            fprintf("False Positive: %d\n",FP);
            fprintf("False Negative: %d\n",FN);
        end
        recall=TP/(TP+FN);
        specificity=TN/(TN+FP);
        precision=TP/(TP+FP);
        negative_predictive_value=TN/(TN+FN);
        fall_out =FP/(FP+TN);
        false_negative_rate=FN/(TP+FN);
        false_discovery_rate =FP/(TP+FP);
        class_accuracy =(TP+TN)/(TP+FP+FN+TN);
        f1=(2*precision*recall)/(precision+recall);
        
        % we set 0/0 = 0, so we replace NaN with 0
        if isnan(f1)
            f1=0;
        end
        if isnan(false_discovery_rate)
            false_discovery_rate=0;
        end
        if isnan(precision)
            precision=0;
        end
        
        total_f1=total_f1+f1;
        if verbose
            fprintf("Recall (TPR): %.4f\n",recall);
            fprintf("Specificity (TPR): %.4f\n",specificity);
            fprintf("Precision (TPR): %.4f\n",precision);
            fprintf("Negative Predictive Value (NPV): %.4f\n",negative_predictive_value);
            fprintf("Fall Out (FPR): %.4f\n", fall_out );
            fprintf("False Negative Rate(FNR): %.4f\n",false_negative_rate);
            fprintf("False Discovery Rate (FDR): %.4f\n",false_discovery_rate);
            fprintf("Class Accuracy (ACC): %.4f\n",class_accuracy);
            fprintf("F1-score: %.4f\n",f1);
            fprintf("\n");
        end
        class_json_str(i)=sprintf('{"Recall":%f,"Specificity":%f,"Negative Predictive Value":%f,"Fall Out":%f,"False negative rate":%f,"False Discovery Rate":%f,"Class Accuracy":%f}', ...
    recall,...
    specificity,...
    negative_predictive_value,...
    fall_out,...
    false_negative_rate,...
    false_discovery_rate,...
    class_accuracy);
    end
    overall_accuracy=trace(confusion_matrix)/sum(sum(confusion_matrix));
    macro_f1=total_f1/n_classes;
    if verbose
        fprintf("Overall Accuracy (ACC): %.4f\n",overall_accuracy);
        fprintf("Macro-F1: %.4f\n",macro_f1);
    end
    eval_json=sprintf('{"ConfusionMatrix": %s, "Classes": [%s],"OverallAccuracy": %f, "MacroF1":%f}',jsonencode(confusion_matrix),join(class_json_str,","),overall_accuracy,macro_f1);
end
