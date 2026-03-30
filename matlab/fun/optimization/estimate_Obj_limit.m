function [J_lim,Obj] = estimate_Obj_limit(p,p_range)
    %判断是否满足限制条件,均满足返回1，有不满足返回0
    %原则上将最容易计算的条件放在最前面
    
    J_lim=0;
    Obj=10^10;
    
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
    
    %稳态误差
    [y,t]=get_step_resp(p(1),p(2),p(3));
    if abs(y(end)-1)>0.01
        return
    end
    
    %超调量
    if max(y)>1.2
        return
    end
    
    J_lim=1;
    %目标函数
    for i=length(t):-1:1
        if abs(y(i)-1)>0.01
            Obj=t(i);
            break
        end
    end
end

function  [y,t]=get_step_resp(P,I,D)
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
    [y,x,t]=step(num,den,t);
end