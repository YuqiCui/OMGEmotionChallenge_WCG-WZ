function [zcr]=CalcuZCR(signal,wlen,inc)
%给一个音频信号矩阵,帧长和帧移，返回一个两列的向量，第一列是短时平均过零率，第二列是帧过零率的均方差
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window 
zcr=zeros(1,2);
    X=enframe(signal,wlen,inc)';    %将信号分帧，存储为矩阵
    fn=size(X,2);
    zcrzcr=zeros(fn,1);
    for j=1:fn
      zcrzcr(j)=sum(X(1:end-1,j).*X(2:end,j)<0);
    end
    zcr(1)=mean(zcrzcr);
    zcr(2)=var(zcrzcr);  %求帧信号的均方差
end

