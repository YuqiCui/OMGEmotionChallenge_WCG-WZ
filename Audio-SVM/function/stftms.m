function d=stftms(x,win,nfft,inc)
%��ʱ����Ҷ�任����,��һ�������ź��Զ���֡���ж�ʱ����Ҷ�任
%���nfft/2+1���У�ÿһ����һ֡���ٸ���Ҷ�任����ֵ
%x�� �ź�
%win: ���ں������֡���ȣ��������Ϊ��֡���ȣ����Զ����ú�������
%nfft������Ҷ�任�Ĵ��ڵĳ��ȣ����ڻ��ߵ���֡��
%Inc��֡��
if length(win)==1               %���win��֡��
    wlen=win;
    win=hanning(wlen);
else
    wlen=length(win);           %���win�Ǹ���Ҷ�任�Ĵ���
end
x=x( : ); win=win( : );
s=length(x);
c=1;
d=zeros((1+nfft/2),1+fix((s-wlen)/inc));

for b=0:inc:(s-wlen)
    u=win.*x((b+1):(b+wlen));
    t=fft(u,nfft);
    d( : ,c)=t(1:(1+nfft/2));
    c=c+1;
end
% d=real(d);%�洢Ϊʵ��
end