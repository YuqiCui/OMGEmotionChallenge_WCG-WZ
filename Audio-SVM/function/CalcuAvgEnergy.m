function AvgEnergy=CalcuAvgEnergy(signal,wlen,inc)
%��һ����Ƶ�źž���,֡����֡�ƣ�����һ����ʱƽ������������
%calculate the short-time average energy
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
win=hanning(wlen);%����������
AvgEnergy=zeros(1,2);
X=enframe(signal,win,inc)';    %���źŷ�֡���洢Ϊ����
fn=size(X,2);%  fn��֡������
En=zeros(fn,1);
for j=1:fn
    u=X( : ,j);                 %ȡ��һ֡
    u2=u.*u;                %�������
    En(j)=sum(u2);        %�õ���һ֡������
end
AvgEnergy(1)=mean(En);
AvgEnergy(2)=var(En);
end