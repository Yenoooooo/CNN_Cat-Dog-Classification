%% 加载数据集
clear all;clc;
imds = imageDatastore('CatDogs', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% 划分数据集 训练集80%，测试集20%
[imgsTrain,imgsTest] = splitEachLabel(imds,0.8,'randomized');

disp(['训练集样本量: ',num2str(numel(imgsTrain.Files))]);
disp(['测试集样本量: ',num2str(numel(imgsTest.Files))]);

%% 仅水平翻转增强
augmenter = imageDataAugmenter('RandXReflection',true);
inputSize = [128 128 3];
augimdsTrain = augmentedImageDatastore(inputSize, imgsTrain, 'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore(inputSize, imgsTest);

%% 网络结构：Block3强化为两层128核
layers = [ ...
    imageInputLayer(inputSize)

    % Block 1
    convolution2dLayer([3 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', 2)

    % Block 2
    convolution2dLayer([3 3], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', 2)

    % Block 3（强化为两层 128 核）
    convolution2dLayer([3 3], 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3], 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', 2)

    % 分类器
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%% 训练参数
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0003, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.0001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose', true, ...
    'VerboseFrequency', 30, ...
    'Plots', 'training-progress');

%% 开始训练
disp('开始训练...');
tic;
net = trainNetwork(augimdsTrain, layers, options);
trainingTime = toc;
disp(['训练耗时: ', num2str(trainingTime/60, '%.2f'), ' 分钟']);

%% 测试集评估
[y_pred, probs] = classify(net, augimdsTest);
accuracy = mean(y_pred == imgsTest.Labels);
disp(['测试集准确率: ', num2str(100*accuracy, '%.2f'), '%']);

%% 混淆矩阵与详细指标
figure('Name', '混淆矩阵', 'Position', [100, 100, 600, 500]);
plotconfusion(imgsTest.Labels, y_pred, '猫狗分类');

C = confusionmat(imgsTest.Labels, y_pred);
disp('混淆矩阵:');
disp(C);

classes = categories(imgsTest.Labels);
for i = 1:length(classes)
    TP = C(i,i); FP = sum(C(:,i))-TP; FN = sum(C(i,:))-TP;
    precision = TP/(TP+FP); recall = TP/(TP+FN);
    f1_score = 2*(precision*recall)/(precision+recall);
    disp([char(classes(i)), ' - 精确率: ', num2str(100*precision,'%.2f'), ...
         '%, 召回率: ', num2str(100*recall,'%.2f'), ...
         '%, F1分数: ', num2str(f1_score,'%.3f')]);
end

disp('===========================================');
disp(['训练时间: ', num2str(trainingTime/60,'%.2f'), ' 分钟']);
disp(['测试准确率: ', num2str(100*accuracy,'%.2f'), '%']);
disp('===========================================');

%% 失败案例分析：显示被误判的猫和狗的图片
% 找出误判的索引
actualLabels = imgsTest.Labels;
wrongIdx = find(y_pred ~= actualLabels);

% 分离出误判为狗的猫（真实猫，预测为狗）和误判为猫的狗（真实狗，预测为猫）
catAsDogIdx = wrongIdx(actualLabels(wrongIdx) == 'cat');
dogAsCatIdx = wrongIdx(actualLabels(wrongIdx) == 'dog');

% 分别取最多3个
numCatShown = min(3, length(catAsDogIdx));
numDogShown = min(3, length(dogAsCatIdx));

% 创建一个图形窗口，2行3列显示
figure('Name', '失败案例分析', 'Position', [100, 100, 900, 600]);

% 显示误判的猫（真实猫，预测为狗）
for i = 1:numCatShown
    subplot(2, 3, i);
    idx = catAsDogIdx(i);
    img = readimage(imgsTest, idx); % 从原始数据读取（未增强）用于显示
    imshow(img);
    trueLabel = char(actualLabels(idx));
    predLabel = char(y_pred(idx));
    confidence = max(probs(idx,:)) * 100;
    title(sprintf('真实: %s → 预测: %s\n置信度: %.1f%%', trueLabel, predLabel, confidence), ...
        'Color', 'r', 'FontSize', 9);
end

% 显示误判的狗（真实狗，预测为猫）
for j = 1:numDogShown
    subplot(2, 3, j+3);
    idx = dogAsCatIdx(j);
    img = readimage(imgsTest, idx);
    imshow(img);
    trueLabel = char(actualLabels(idx));
    predLabel = char(y_pred(idx));
    confidence = max(probs(idx,:)) * 100;
    title(sprintf('真实: %s → 预测: %s\n置信度: %.1f%%', trueLabel, predLabel, confidence), ...
        'Color', 'r', 'FontSize', 9);
end

% 若图片不足3个，剩余的subplot留空
if numCatShown < 3
    for k = numCatShown+1:3
        subplot(2,3,k); axis off;
    end
end
if numDogShown < 3
    for k = numDogShown+1:3
        subplot(2,3,k+3); axis off;
    end
end