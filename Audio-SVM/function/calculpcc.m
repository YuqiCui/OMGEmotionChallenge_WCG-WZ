function lpcc=calculpcc(signal,wlen,inc)
%���������źž��󣬼�����mfcc���������ؾ�����=�����źŵ���������=MFCC��������
%��ʼ�����������������ÿһ���ź�
lpcc=zeros(1,24); %ÿһ���źŷ���12ά����
temp=lpcc_m(signal,wlen,inc);%���ص���һ��֡��*12�ľ���
lpcc(1: 24)=temp;

end

function ppp=lpcc_m(x,framesize,inc)
%����һ����Ƶ�źţ����źŽ��з�֡��Ȼ����ÿ֡��lpccϵ����������þ�ֵ

%Ԥ�����˲���
xx=double(x);
xx=filter([1-0.9375],1,xx);

%�����źŷ�֡
xx=enframe(xx,framesize,inc);
m=zeros(size(xx,1),12);

for i=1:size(xx,1) %����ÿһ֡
    X=xx(i,:);
    [ar,~]=lpc(X,12);%x��һ֡�����ݣ�p������Ԥ��Ľ���������ȡ12
    temp=lpc2lpccm(ar,12,12);%���lpccϵ����������
    m(i,:)=temp;
end
m(find(isnan(m)==1)) = 0;
ppp(1:12)=mean(m,1);%���ÿһ�еľ�ֵ
ppp(13:24)=var(m);
ppp(find(isnan(ppp)==1)) = 0;
end

function lpcc=lpc2lpccm(ar,n_lpc,n_lpcc)
lpcc=zeros(n_lpcc,1);
lpcc(1)=ar(1);
for n=2:n_lpc
    lpcc(n)=ar(n);
    for l=1:n-1
        lpcc(n)=lpcc(n)+ar(l)*lpcc(n-l)*(n-l)/n;
    end
end
for n=(n_lpc+1):n_lpcc
    lpcc(n)=0;
    for l=1:n_lpc
        lpcc(n)=lpcc(n)+ar(l)*lpcc(n-l)*(n-l)/n;
    end
end
lpcc=-lpcc;
end

