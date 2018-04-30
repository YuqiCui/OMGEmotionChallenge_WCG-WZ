%% Initialization
clear ; close all; clc

%% ==================== Part 1: Extract Audio ====================
% inputfile='D:\Desktop\OMG-competition\data\omg_train\';

% Batch processing of train and validation data 
% omg=importdata('data\omg_TestVideo.mat');
% hash=table2array(omg(2:end,4));
% utterance=table2array(omg(2:end,5));
% hash=omg.textdata( 2:end ,4);
% utterance=omg.textdata( 2:end ,5);
% FeatureMatrix=zeros(length(hash),76);%the length of the audio is unfixed,so we can only calculate the feature one by one
% for i=1:length(hash)   %205
%     try
%     filepath=strcat(inputfile,hash{i},'\',utterance{i});  %根据hash值，找到相应的文件位置路径
%         if exist(filepath,'file')~=0
%             [input_file, Fs] = audioread(filepath);
%             if size(input_file,2)==1
%                 input_file=input_file( : ,1);
%                 fprintf('%d, %s, %s\n',i,hash{i},utterance{i});
%             end
%             FeatureMatrix(i, : )=FeatureExtraction(input_file,Fs);
%         end
%     catch
%         fprintf('%d\n',i);
%     end
% end

%%Test data
inputfile='D:\Desktop\OMG-competition\data\omg_Test_cor\';
omg=importdata('data\testidx.mat');
hash=omg( : ,1);
utterance=omg( :,2);
FeatureMatrix=zeros(length(hash),76);%the length of the audio is unfixed,so w   e can only calculate the feature one by one
for i=1:length(hash)   %205
    try
        filepath=char(strcat(inputfile,hash{i},'\',utterance{i})); %根据hash值，找到相应的文件位置路径
        if exist(filepath,'file')~=0
            [input_file, Fs] = audioread(filepath);
            if size(input_file,2)==2
                input_file=input_file( : ,1);
            end
            FeatureMatrix(i, : )=FeatureExtraction(input_file,Fs);
            fprintf('%d, %s, %s\n',i,hash{i},utterance{i});
        end
    catch
        fprintf('error!%d\n',i);
    end
end

save featureMatrix_test FeatureMatrix;



