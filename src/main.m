clear; clc
close all
tic
addpath(genpath(pwd))


windowLength = 0.5;    % window length
stepLength = 0.02;      % step length

% DIVIDE THE DATA INTO 70 % FOR TRAINING AND 30 % FOR TESTING %

%path to dataset & list of all .ogg files
cow_path = [pwd,'/ESC-50-master/104 - Cow/*.ogg'];
clapping_path = [pwd, '/ESC-50-master/303 - Clapping/*.ogg'];
engine_path = [pwd, '/ESC-50-master/505 - Engine/*.ogg'];

%struct of all files for every classes
cow_files = dir(cow_path);
clapping_files = dir(clapping_path);
engine_files = dir(engine_path);

%substructs for train and test
cow_train = cow_files(1:round(0.7 * numel(cow_files)));
clapping_train = clapping_files(1:round(0.7 * numel(clapping_files)));
engine_train = engine_files(1:round(0.7 * numel(engine_files)));

cow_test = cow_files(round(0.7*numel(cow_files))+1:end);
clapping_test = clapping_files(round(0.7*numel(clapping_files))+1:end);
engine_test = engine_files(round(0.7*numel(engine_files))+1:end);

% EXTRACT ALL THE AUDIO FEATURES (TIME AND FREQUENCY DOMAIN) %

cow_trainFeats = [];
clapping_trainFeats = [];
engine_trainFeats = [];
cow_trainTime = [];
clapping_trainTime = [];
engine_trainTime = [];
w1 = waitbar(0, 'Extracting train features...');

% Extracting train features

for i=1:length(cow_train)
    
    [C,S,R,ceps]=frequency_features(cow_train(i).name, windowLength, stepLength);
    [E, EE, Z]=timedomainFeats(cow_train(i).name,windowLength,stepLength);
    waitbar(i / length(cow_train),w1);
    tmp = [C' S' R' ceps'];
    cow_trainFeats = [cow_trainFeats; tmp];
    tmp = [E' EE' Z'];
    cow_trainTime = [cow_trainTime; tmp];
end

for i=1:length(clapping_train)
    
    [C,S,R,ceps]=frequency_features(clapping_train(i).name,windowLength,stepLength);
    [E, EE, Z]=timedomainFeats(clapping_train(i).name,windowLength,stepLength);
    waitbar(i / length(clapping_train),w1);
    tmp = [C' S' R' ceps'];
    clapping_trainFeats = [clapping_trainFeats; tmp];
    tmp = [E' EE' Z'];
    clapping_trainTime = [clapping_trainTime; tmp];
end

for i=1:length(engine_train)
    [C,S,R,ceps]=frequency_features(engine_train(i).name,windowLength,stepLength);
    [E, EE, Z]=timedomainFeats(engine_train(i).name,windowLength,stepLength);
    waitbar(i / length(engine_train),w1);
    tmp = [C' S' R' ceps'];
    engine_trainFeats = [engine_trainFeats; tmp];
    tmp = [E' EE' Z'];
    engine_trainTime = [engine_trainTime; tmp];
end

close(w1);
alltrainFeats = [cow_trainFeats; clapping_trainFeats; engine_trainFeats];
alltrainTime = [cow_trainTime; clapping_trainTime; engine_trainTime];

% Extracting test features

w2 = waitbar(0, 'Extracting test features...');

cow_testFeats = [];
clapping_testFeats = [];
engine_testFeats = [];
cow_testTime = [];
clapping_testTime = [];
engine_testTime = [];

for i=1:length(cow_test)
    [C,S,R,ceps]=frequency_features(cow_test(i).name,windowLength,stepLength);
    [E, EE, Z]=timedomainFeats(cow_test(i).name,windowLength,stepLength);
    waitbar(i / length(cow_test),w2);
    tmp = [C' S' R' ceps'];
    cow_testFeats = [cow_testFeats; tmp];
    tmp = [E' EE' Z'];
    cow_testTime = [cow_testTime; tmp];
end

for i=1:length(clapping_test)
    [C,S,R,ceps]=frequency_features(clapping_test(i).name,windowLength,stepLength);
    [E, EE, Z]=timedomainFeats(clapping_test(i).name,windowLength,stepLength);
    waitbar(i / length(clapping_test),w2);
    tmp = [C' S' R' ceps'];
    clapping_testFeats = [clapping_testFeats; tmp];
    tmp = [E' EE' Z'];
    clapping_testTime = [clapping_testTime; tmp];
end

for i=1:length(engine_test)
    [C,S,R,ceps]=frequency_features(engine_test(i).name,windowLength,stepLength);
    [E, EE, Z]=timedomainFeats(engine_test(i).name,windowLength,stepLength);
    waitbar(i / length(engine_test),w2);
    tmp = [C' S' R' ceps'];
    engine_testFeats = [engine_testFeats; tmp];
    tmp = [E' EE' Z'];
    engine_testTime = [engine_testTime; tmp];
end

close(w2);

% Concatenate features of all classes (train / test)
alltestFeats = [cow_testFeats; clapping_testFeats; engine_testFeats];
alltestTime = [cow_testTime; clapping_testTime; engine_testTime];
allFeats = [alltrainFeats; alltestFeats];
allTime = [alltrainTime; alltestTime];
features = [allFeats allTime];

cowTrain = [cow_trainFeats cow_trainTime];
cowTest = [cow_testFeats cow_testTime];
cowComplete = [cowTrain; cowTest];

clappingTrain = [clapping_trainFeats clapping_trainTime];
clappingTest = [clapping_testFeats clapping_testTime];
clappingComplete = [clappingTrain; clappingTest];

engineTrain = [engine_trainFeats engine_trainTime];
engineTest = [engine_testFeats engine_testTime];
engineComplete = [engineTrain; engineTest];

alltrain = [cowTrain; clappingTrain; engineTrain];
alltest = [cowTest; clappingTest; engineTest];

% Apply normalization on allFeats
mn = mean(allFeats);
st = std(allFeats);
allFeats = (allFeats - repmat(mn, size(allFeats,1),1))./repmat(st,size(allFeats,1),1);

%Apply normalization on allTime
mn = mean(allTime);
st = std(allTime);
allTime = (allTime - repmat(mn, size(allTime,1),1))./repmat(st,size(allTime,1),1);

%Apply normalization on features
mn = mean(features);
st = std(features);
features = (features - repmat(mn, size(features,1),1))./repmat(st,size(features,1),1);

% APPLY THE PCA %

[coeff,score,latent,tsquared,explained] = pca(features);
explained

% Plotting PCA
S = [];
C = [repmat([1 0 0],length(cowComplete),1); repmat([0 1 0],length(clappingComplete),1); repmat([0 0 1],length(engineComplete),1)];

% Define color RGB

scatter3(score(:,1),score(:,2),score(:,3),S,C)
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')
sgtitle('features after PCA')

% DEFINE COEFFICIENTS WITH 80% OF VARIANCE

cumulativeExplained = cumsum(explained);
numCoeff = find(cumulativeExplained >= 80,1,'first');

disp(numCoeff)

% TRAIN THE KNN ~ TIME DOMAIN ONLY

% Create labels
labelcowTime = repmat(1,length(cow_trainTime),1);
labelclappingTime = repmat(2,length(clapping_trainTime),1);
labelengineTime = repmat(3,length(engine_trainTime),1);

% Create groundtruth
labelcowTime1 = repmat(1,length(cow_testTime),1);
labelclappingTime1 = repmat(2,length(clapping_testTime),1);
labelengineTime1 = repmat(3,length(engine_testTime),1);

ground_truthTime = [labelcowTime1; labelclappingTime1; labelengineTime1];

disp('kNN only time domain')
all_labels = [labelcowTime; labelclappingTime; labelengineTime];

% Apply normalization on alltrainTime
mn = mean(alltrainTime);
st = std(alltrainTime);
alltrainTime = (alltrainTime - repmat(mn,size(alltrainTime,1),1))./repmat(st,size(alltrainTime,1),1);
% Apply normalization on alltestTime
mn = mean(alltestTime);
st = std(alltestTime);
alltestTime = (alltestTime - repmat(mn,size(alltestTime,1),1))./repmat(st,size(alltestTime,1),1);

% known Nearest Neighbor

k = [1 2 5 10 20 50 100];
rate = [];
for kk=1:length(k)
    disp(['set-up the kNN... number of neighbors: ',mat2str(k(kk))])
    Mdl = fitcknn(alltrainTime,all_labels','NumNeighbors',k(kk));
    
    % Test the kNN
    predicted_label = predict(Mdl,alltestTime);
    
    % Measure the performance
    correct = 0;
    for i=1:length(predicted_label)
        if predicted_label(i)==ground_truthTime(i)
            correct=correct+1;
        end
    end
    disp('recognition rate:')
    rate(kk) = (correct/length(predicted_label))*100
end

figure;
plot(k,rate)
xlabel('k')
ylabel('recognition rate (%)')
sgtitle('kNN ~ Time Domain')
grid on
% Print the maximum recognition rate (time domain)
[a,b]=max(rate);
disp('--------results---------')
disp(['the maximum recognition rate (time domain only) is ',mat2str(a)])
disp(['and it is achieved with ',mat2str(k(b)),' nearest neighbors'])

% TRAIN THE KNN ~ ONLY FEATURES DOMAIN

% Create labels
labelcowFeats = repmat(1,length(cow_trainFeats),1);
labelclappingFeats = repmat(2,length(clapping_trainFeats),1);
labelengineFeats = repmat(3,length(engine_trainFeats),1);

% Create groundtruth
labelcowFeats1 = repmat(1,length(cow_testFeats),1);
labelclappingFeats1 = repmat(2,length(clapping_testFeats),1);
labelengineFeats1 = repmat(3,length(engine_testFeats),1);

ground_truthFeats = [labelcowFeats1; labelclappingFeats1; labelengineFeats1];

disp('kNN only features domain')
all_labels2 = [labelcowFeats; labelclappingFeats; labelengineFeats];

% Apply normalization on alltrainFeats
mn = mean(alltrainFeats);
st = std(alltrainFeats);
alltrainFeats = (alltrainFeats - repmat(mn,size(alltrainFeats,1),1))./repmat(st,size(alltrainFeats,1),1);
% Apply normalization on alltestFeats
mn = mean(alltestFeats);
st = std(alltestFeats);
alltestFeats = (alltestFeats - repmat(mn,size(alltestFeats,1),1))./repmat(st,size(alltestFeats,1),1);

% known Nearest Neighbor

k = [1 2 5 10 20 50 100];
rate = [];
for kk=1:length(k)
    disp(['set-up the kNN... number of neighbors: ',mat2str(k(kk))])
    Mdl = fitcknn(alltrainFeats,all_labels2','NumNeighbors',k(kk));
    
    % Test the kNN
    predicted_label = predict(Mdl,alltestFeats);
    
    % Measure the performance
    correct = 0;
    for i=1:length(predicted_label)
        if predicted_label(i)==ground_truthFeats(i)
            correct=correct+1;
        end
    end
    disp('recognition rate:')
    rate(kk) = (correct/length(predicted_label))*100
end

figure;
plot(k,rate)
xlabel('k')
ylabel('recognition rate (%)')
sgtitle('kNN ~ Features Domain')
grid on
% Print the maximum recognition rate (features domain)
[a,b]=max(rate);
disp('--------results---------')
disp(['the maximum recognition rate (features domain only) is ',mat2str(a)])
disp(['and it is achieved with ',mat2str(k(b)),' nearest neighbors'])

% Train the kNN ~ features together

% Create labels
labelcowComplete = repmat(1,length(cowTrain),1);
labelclappingComplete = repmat(2,length(clappingTrain),1);
labelengineComplete = repmat(3,length(engineTrain),1);

% Create groundtruth
labelcowComplete1 = repmat(1,length(cowTest),1);
labelclappingComplete1 = repmat(2,length(clappingTest),1);
labelengineComplete1 = repmat(3,length(engineTest),1);

ground_truthComplete = [labelcowComplete1; labelclappingComplete1; labelengineComplete1];

disp('kNN altogether')
all_labels3 = [labelcowComplete; labelclappingComplete; labelengineComplete];

% Apply normalization on alltrain
mn = mean(alltrain);
st = std(alltrain);
alltrain = (alltrain - repmat(mn,size(alltrain,1),1))./repmat(st,size(alltrain,1),1);

%Apply normalization on alltest
mn = mean(alltest);
st = std(alltest);
alltest = (alltest - repmat(mn,size(alltest,1),1))./repmat(st,size(alltest,1),1);

% known Nearest Neighbor

k = [1 2 5 10 20 50 100];
rate = [];
for kk=1:length(k)
    disp(['set-up the kNN... number of neighbors: ',mat2str(k(kk))])
    Mdl = fitcknn(alltrain,all_labels3','NumNeighbors',k(kk));
    
    % Test the kNN
    predicted_label = predict(Mdl,alltest);
    
    % Measure the performance
    correct = 0;
    for i=1:length(predicted_label)
        if predicted_label(i)==ground_truthComplete(i)
            correct=correct+1;
        end
    end
    disp('recognition rate:')
    rate(kk) = (correct/length(predicted_label))*100
end

figure;
plot(k,rate)
xlabel('k')
ylabel('recognition rate (%)')
sgtitle('kNN ~ Altogether')
grid on
% Print the maximum recognition rate (features together)
[a,b]=max(rate);
disp('--------results---------')
disp(['the maximum recognition rate (features together) is ',mat2str(a)])
disp(['and it is achieved with ',mat2str(k(b)),' nearest neighbors'])

toc





















