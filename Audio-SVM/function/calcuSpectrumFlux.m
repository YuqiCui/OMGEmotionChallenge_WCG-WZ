function SpecFlux=calcuSpectrumFlux(signal,wlen,inc)
%���������źž��󣬷���һά����Spectrum Flux
num=size(signal, 2);%��ʼ�����������
win=hanning(wlen);%����������
SpecFlux=zeros(num,1);


for i=1:num;           %����ÿһ����Ƶ�ź�
   x=signal( : ,i);       %���һ����Ƶ�ź�
    X=enframe(x,win,inc)';    %���źŷ�֡���洢Ϊ����
    [m,n]=size(X);                % ��Ϊÿ֡�ĳ��ȣ���Ϊ֡��
   
    Y=zeros(m/2+1,n) ;              %�½�һ�����������洢����Ҷ�任��Ľ��
    
   for ii=1:n    %����ÿһ֡����
       %���ȶ�ÿһ֡���ݽ���һ����ɢ�任
       u=X( : , ii);
       nfft=wlen;
       Y( : ,ii)=computeDFTviaFFT(u,wlen,nfft);   %��ÿһ֡���ݽ���һ����ɢ����Ҷ�任������Y�У���������Ԫ�ظ���Ӧ����nfft/2+1
       [yraw,ycol]=size(Y);
   end
    
        M=zeros(yraw,ycol-1);
   %��������ֵ
   for k=2:n  %����ÿһ֡
       M( : ,ycol-1)=power((log(Y( : ,k))-log(Y( : ,k-1))),2);  
   end
   A=ones(ycol-1,1);
   B=ones(1,yraw);
   SpecFlux(i)=(B*(M*A))/((ycol-1)*yraw);
 
end
end

function [Xx] = computeDFTviaFFT(xin,nx,nfft)
% Use FFT to compute raw STFT and return the F vector.
%nx: �źų��ȣ�֡��
%nfft�� ����Ҷ�任�Ĵ��ڳ���
% Handle the case where NFFT is less than the segment length, i.e., "wrap"
% the data as appropriate.
xin_ncol = size(xin,2);          
xw = zeros(nfft,xin_ncol);   
if nx > nfft                         %�ж���һ��֡���͸���Ҷ�任���ڳ��ȵĹ�ϵ  %���֡�����ڴ��ڳ�
    for j = 1:xin_ncol
        xw(:,j) = datawrap(xin(:,j),nfft);
    end
else
    xw = xin;
end
Xx1 = abs(fft(xw,nfft));
m=size(Xx1,2);
mm=m/2+1;
Xx=Xx1(1:mm);
%f = psdfreqvec('npts',nfft,'Fs',Fs);

end

