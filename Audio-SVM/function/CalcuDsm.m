function Dsm=CalcuDsm(signal,wlen,inc)
%��һ����Ƶ�źž���,֡����֡��,�����ʣ�����һ��һ�е��������������Delta spectrum magnitude
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
Dsm=zeros(1,2);
nfft=wlen;%���ÿ��ٸ���Ҷ�任�Ĵ��ڳ��Ⱥ�֡����ͬ
M=stftms(signal,wlen,nfft,inc);  %��ÿһ���ź����뵽��ʱ����Ҷ�任�����н��ж�ʱ����Ҷ�任
[m,n]=size(M);%��ʼ��M������к���
%����ÿһ֡��Delta spectrum magnitude���������
add=zeros(1,n);
for j=1:n
    u=M( : ,j);    %ȡ��һ֡������
    for x1=1:(m-1)
        add(j)=add(j)+abs(      abs(  u(x1)  )  -    abs(u(x1+1))     );
    end
end
Dsm(1)=mean(add);
Dsm(2)=var(add);
Dsm=real(Dsm);
end