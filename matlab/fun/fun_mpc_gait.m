function gait=fun_mpc_gait(pref,zc,dt)
    % 模型预测控制MPC
    % 输入参数说明：pref是ZMP参考位置（期望落脚点，一维随时间变化数据），zc是质心高度，dt是控制时间步长
    % 输出参数包括：时间序列，参考ZMP位置，实际ZMP位置，实际质心位置（根据ZMP模型预测控制计算得到）
    g=9.8;
    T=(length(pref)-1)*dt;
    t=0:dt:T;
    
    N=5000; %考虑未来的步数
    pref=[pref,ones(1,N)*pref(end)];
    
    Q=1000; %权重系数
    R=1; %权重系数
    A=[1,dt,dt^2/2;0,1,dt;0,0,1];
    B=[dt^3/6,dt^2/2,dt]';
    C=[1,0,-zc/g];
    Ap=[1,C*A; zeros(3,1),A];
    Bp=[C*B;B];
    Cp=[1 0 0 0];
    QQ=Cp'*Q*Cp;
    [Kp,Pp,~]=dlqr(Ap,Bp,QQ,R,0);
    
    fp=zeros(1,N);
    for i=1:N
        fp(i)=pinv(R+Bp'*Pp*Bp)*Bp'*((Ap-Bp*Kp)')^(i-1)*Cp'*Q;
    end
    
    u=zeros(1,length(t));   %输入量dddot(x)
    du=zeros(1,length(t));
    x=zeros(3,length(t));  %质心位置、速度、加速度的列向量
    dx=zeros(3,length(t));
    p=zeros(1,length(t));  %ZMP位置
    xs=[p;dx];
    
    for i=1:length(t)
        du(i)=-Kp*xs(:,i)+fp*[pref(i+1:i+N)]';
        xs(:,i+1)=Ap*xs(:,i)+Bp*du(i);
        dx(:,i+1)=xs(2:4,i+1);
        x(:,i+1)=x(:,i)+dx(:,i+1);
        p(i)=Cp*xs(:,i);
    end
    
    plot(t,pref(1:length(t)));
    hold on
    plot(t,p);
    hold on
    plot(t,x(1,1:length(t)))
    hold on
    legend('期望ZMP','实际ZMP','质心位置')
    
    gait=[t',pref(1:length(t))',p',x(1,1:length(t))'];