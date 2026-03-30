function J_lim = estimate_limit(p,p_range)
    %判断是否满足限制条件,均满足返回1，有不满足返回0
    %原则上将最容易计算的条件放在最前面
    
    J_lim=0;
    
    %参数范围
    n=length(p);
    JJ=zeros(1,n);
    for i=1:n
        if p(i)>=p_range(1,i)&&p(i)<=p_range(2,i)
            JJ(i)=1;
        end
    end
    if sum(JJ)<n
        return
    end
    
    %阶跃稳态误差
    [y1,y2,t]=get_resp(p(1),p(2),p(3));
    if abs(y1(end)-1)>0.01
        return
    end
    
    %阶跃响应时间
    for i=length(t):-1:1
        if abs(y1(i)-1)>0.01
            T=t(i);
            break
        end
    end
    if T>1
        return
    end
    J_lim=1;
end


function  [y1,y2,t]=get_resp(P,I,D)

    N=100;
    m=1000*0.05*0.05*0.5;   %杆的质量
    J=m*(0.5^2+0.05^2)/12+m*0.25^2;  %杆的转动惯量
    %PID传递函数
    num1=[P+N*D,P*N+I,I*N];
    den1=[1,N,0];
    %物理系统的传递函数
    num2=[1];
    den2=[J,0,0];
    %PID串联在物理系统
    [numg,deng]=series(num1,den1,num2,den2);
    %增加反馈
    numf=[1];
    denf=[1];
    [num,den]=feedback(numg,deng,numf,denf);
    
    t=0:0.001:10;
    y1=step(num,den,t);  %阶跃响应
    r=sin(t);
    y2=lsim(num,den,r,t); %正弦响应
end