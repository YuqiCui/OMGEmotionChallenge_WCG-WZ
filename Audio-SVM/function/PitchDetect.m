function pitch=PitchDetect(signal,fs,wlen,inc)
%ʹ�û��ڵ��׷��Ի������ڽ��м��
%�����������źž������һ�л�������ֵ
pitch=zeros(1,2);
    signal=signal-mean(signal);   %��ȥֱ������
    y1=enframe(signal,wlen,inc)'; %��֡
    signal=signal/max(abs(signal));  %��ֵ��һ��
    y=enframe(signal,wlen,inc)'; %��֡
    fn=size(y,2); %ȡ����֡��
%     time=(0:length(x)-1)/fs;  %����ʱ������
%     frameTime=frame2time(fn,wlen,inc,fs);%�����֡��Ӧ��ʱ������
    T1=0.05;  %�������Ĳ���
    [~,~,SF,~]=pitch_vadl(y,fn,T1); %�����Ķ˵���
%     lmin=fix(fs/500);   %�������ڵ���Сֵ
    %lmax=fix(fs/60);   %�������ڵ����ֵ
    period=zeros(1,fn);   %�������ڵĳ�ʼ��
    for k=1:fn
        if SF(k)==1      %�Ƿ����л�֡��
            y1=y( : ,k).*hamming(wlen);    %ȡ��һ֡���ݼӴ�����
            xx=fft(y1);
            a=2*log(abs(xx)+eps);
            b=ifft(a);
            [~,Lc(k)]=max(b(1:200));     %��lmin��lmaxzhong ��Ѱ�����ֵ
           % period(k)=Lc(k)+lmin-1;  %������������
           period(k)=Lc(k)+1-1; 
        end
    end
    pitch(1)=mean(period);
    pitch(2)=var(period);
end

%�����رȵķ������ж˵���
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

%���������ʼ�ĺͽ�����ʱ�䣬�Լ���������ĳ��ȡ�
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
%����ÿ�黰�εĳ���
for i=1:k
    soundsegment(i).duration=soundsegment(i).end-soundsegment(i).begin+1;
end
end
