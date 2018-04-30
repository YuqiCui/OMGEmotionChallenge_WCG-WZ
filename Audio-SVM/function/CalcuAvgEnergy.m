function AvgEnergy=CalcuAvgEnergy(signal,wlen,inc)
%给一个音频信号矩阵,帧长和帧移，返回一个短时平均能量的向量
%calculate the short-time average energy
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
win=hanning(wlen);%给出海宁窗
AvgEnergy=zeros(1,2);
X=enframe(signal,win,inc)';    %将信号分帧，存储为矩阵
fn=size(X,2);%  fn是帧的数量
En=zeros(fn,1);
for j=1:fn
    u=X( : ,j);                 %取出一帧
    u2=u.*u;                %求出能量
    En(j)=sum(u2);        %得到这一帧的能量
end
AvgEnergy(1)=mean(En);
AvgEnergy(2)=var(En);
end