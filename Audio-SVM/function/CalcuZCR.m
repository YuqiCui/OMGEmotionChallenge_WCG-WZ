function [zcr]=CalcuZCR(signal,wlen,inc)
%��һ����Ƶ�źž���,֡����֡�ƣ�����һ�����е���������һ���Ƕ�ʱƽ�������ʣ��ڶ�����֡�����ʵľ�����
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window 
zcr=zeros(1,2);
    X=enframe(signal,wlen,inc)';    %���źŷ�֡���洢Ϊ����
    fn=size(X,2);
    zcrzcr=zeros(fn,1);
    for j=1:fn
      zcrzcr(j)=sum(X(1:end-1,j).*X(2:end,j)<0);
    end
    zcr(1)=mean(zcrzcr);
    zcr(2)=var(zcrzcr);  %��֡�źŵľ�����
end

