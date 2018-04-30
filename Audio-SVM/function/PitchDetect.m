function pitch=PitchDetect(signal,fs,wlen,inc)
%使用基于倒谱法对基因周期进行检测
%输入组语音信号矩阵，输出一列基音周期值
pitch=zeros(1,2);
    signal=signal-mean(signal);   %消去直流分量
    y1=enframe(signal,wlen,inc)'; %分帧
    signal=signal/max(abs(signal));  %幅值归一化
    y=enframe(signal,wlen,inc)'; %分帧
    fn=size(y,2); %取得总帧数
%     time=(0:length(x)-1)/fs;  %计算时间坐标
%     frameTime=frame2time(fn,wlen,inc,fs);%计算各帧对应的时间坐标
    T1=0.05;  %基音检测的参数
    [~,~,SF,~]=pitch_vadl(y,fn,T1); %基音的端点检测
%     lmin=fix(fs/500);   %基音周期的最小值
    %lmax=fix(fs/60);   %基音周期的最大值
    period=zeros(1,fn);   %基音周期的初始化
    for k=1:fn
        if SF(k)==1      %是否在有话帧中
            y1=y( : ,k).*hamming(wlen);    %取来一帧数据加窗函数
            xx=fft(y1);
            a=2*log(abs(xx)+eps);
            b=ifft(a);
            [~,Lc(k)]=max(b(1:200));     %在lmin和lmaxzhong 中寻找最大值
           % period(k)=Lc(k)+lmin-1;  %给出基音周期
           period(k)=Lc(k)+1-1; 
        end
    end
    pitch(1)=mean(period);
    pitch(2)=var(period);
end

%用能熵比的方法进行端点检测
function [voiceseg,vosl,SF,Ef]=pitch_vadl(y,fn,T1,miniL)
if nargin<4,miniL=10;end
if size(y,2)~=fn,y=y';end
wlen=size(y,1);
for i=1:fn
    Sp=abs(fft(y( : ,i)));
    Sp=Sp(1:wlen/2+1);
    Esum(i)=sum(Sp.*Sp);
    prob=Sp/(sum(Sp));
    H(i)=-sum(prob.*log(prob+eps));
end
hindex=find(H<0.1);
H(hindex)=max(H);
Ef=sqrt(1+abs(Esum./H));
Ef=Ef/max(Ef);

zindex=find(Ef>=T1);
zseg=findSegment(zindex);
zsl=length(zseg);
j=0;
SF=zeros(1,fn);
for k=1:zsl
    if zseg(k).duration>=miniL
        j=j+1;
        in1=zseg(k).begin;
        in2=zseg(k).end;
        voiceseg(j).begin=in1;
        voiceseg(j).end=in2;
        voiceseg(j).duration=zseg(k).duration;
        SF(in1:in2)=1;
    end
end
vosl=length(voiceseg);
end

%检测语音开始的和结束的时间，以及这段语音的长度。
function soundsegment=findSegment(express)
if express(1)==0
    voicedIndex=find(express);
else
    voicedIndex=express;
end
soundsegment=[];
k=1;
soundsegment(k).begin=voicedIndex(1);
for i=1:length(voicedIndex)-1
    if voicedIndex(i+1)-voicedIndex(i)>1
        soundsegment(k).end=voicedIndex(i);
        soundsegment(k+1).begin=voicedIndex(i+1);
        k=k+1;
    end
end
soundsegment(k).end=voicedIndex(end);
%计算每组话段的长度
for i=1:k
    soundsegment(i).duration=soundsegment(i).end-soundsegment(i).begin+1;
end
end
