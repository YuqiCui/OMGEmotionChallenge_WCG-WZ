function SpecCentroid=CalcuSpeCent(signal,fs,wlen,inc)
%��һ����Ƶ�źž���,֡����֡��,�����ʣ�����һ��һ�е��������������spectrum centroid
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
SpecCentroid=zeros(1,2);
nfft=wlen;%���ÿ��ٸ���Ҷ�任�Ĵ��ڳ��Ⱥ�֡����ͬ
M=stftms(signal,wlen,nfft,inc);  %��ÿһ���ź����뵽��ʱ����Ҷ�任�����н��и���Ҷ�任
[m,n]=size(M);%��ʼ��M������к���
%����ÿһ֡��spectrum centroid���������
add=zeros(1,n);
for j=1:n
    u=M( : ,j);    %ȡ��һ֡������
    t=fs/(m-1); %t=1/T;
    A=0:(m-1);
    B=A';
    C=B*t;
    D=sum(abs(u.*u));
    if D==0
        add(j)=0;
    else
        add(j)=sum(abs(u.*u).*C)/sum(abs(u.*u));
    end
end
SpecCentroid(1)=mean(add);
SpecCentroid(2)=var(add);
SpecCentroid=real(SpecCentroid);
end



