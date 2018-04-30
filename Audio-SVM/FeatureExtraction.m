function features=FeatureExtraction(AudioSignal,Fs)
features=zeros(1,76);
wlen=200;%给出帧长
inc=80;%给出帧移

% %short-time Average Energy 
%mean+var(2)
features(1:2)=CalcuAvgEnergy(AudioSignal,wlen,inc);

% %short-time zero crossing rate
%mean+var(2)
 features(3:4)=CalcuZCR(AudioSignal,wlen,inc);

% %%Time frequency
% %Spectral centroid
%mean+var(2)
 features(5:6)=CalcuSpeCent(AudioSignal,Fs,wlen,inc);

 % %Band energy radio
%mean+var(2)
features( : ,7:8)=CalcuBer(AudioSignal,wlen,inc);

%Delta spectrum magnitude8
% mean+var(2)
 features( : ,9:10)=CalcuDsm(AudioSignal,wlen,inc);
 
 % %silence ratio
 features( : ,11)=calcuSilenceRatio(AudioSignal,Fs);
 
 % formants
 features( : ,12:16)=CalcuFormants(AudioSignal,Fs,wlen,inc);
 
 %Pitch
%mean+var(2)
features( : ,17:18)=PitchDetect(AudioSignal,Fs,wlen,inc);

% %MFCC  12阶+12阶差分+12阶数据的方差
 features( : ,19:54)=calcumfcc(AudioSignal,Fs,wlen,inc);
 
  %LPCC features
 features( : ,55:78)=calculpcc(AudioSignal,wlen,inc);
 features( : ,55)=[];
 features( : ,66)=[];


end