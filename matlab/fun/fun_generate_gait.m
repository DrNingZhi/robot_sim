function gait=fun_generate_gait()
    %开始-步行-停止模式，获得重心位置和足底位置随时间的变化
    W0=0.2; %胯宽
    zc=1; %质心高度
    Tstep=0.5; %半步周期
    double_support_ratio=0.1; %双支撑时间占比
    Lstep=0.5; %步长
    Nstep=15; %半步数
    dt=0.001; %时间步长
    H=0.1; %抬脚高度
    
    Ttot=Tstep*(Nstep+2); %前后各增加一个周期用于稳定
    t=0:dt:Ttot;
    
    %ZMP
    pref=zeros(length(t),3); %第一列是时间，第二列是ZMP的x，第三列是ZMP的y
    pref(:,1)=t';
    for i=2:Nstep+1
        if mod(i,2)==0 %单数步，右腿支撑
            pref((i-1)*round(Tstep/dt)+2:i*round(Tstep/dt),3)=-W0/2;
        else
            pref((i-1)*round(Tstep/dt)+2:i*round(Tstep/dt),3)=W0/2;
        end
        pref((i-1)*round(Tstep/dt)+2:i*round(Tstep/dt)+1,2)=(i-2)*Lstep;
    end
    pref((Nstep+1)*round(Tstep/dt)+2:end,2)=(i-2)*Lstep;
    
    %质心位置计算
    gaitx=fun_mpc_gait(pref(:,2)',zc,dt);
    gaity=fun_mpc_gait(pref(:,3)',zc,dt);
    
    %铺脚印
    Lfoot=zeros(length(t),4); %左脚的txyz
    Lfoot(:,1)=t';
    Lfoot(:,2)=10^10;
    Lfoot(:,3)=W0/2;
    Rfoot=zeros(length(t),4); %左脚的txyz
    Rfoot(:,1)=t';
    Rfoot(:,2)=10^10;
    Rfoot(:,3)=-W0/2;
    doubleN=round(double_support_ratio*(Tstep/dt)/2);
    %初始双支撑时间
    Lfoot(1:round(Tstep/dt)+doubleN,2)=pref(1,2);
    Rfoot(1:round(Tstep/dt)+doubleN,2)=pref(1,2);
    for i=1:length(t)-1
        if pref(i,3)==0&&pref(i+1,3)>0 %左脚落地时刻
            if i+1-doubleN<0
                Lfoot(:,2)=pref(i+1,2);
            else
                Lfoot(i+1-doubleN:end,2)=pref(i+1,2);
            end
        end
        if pref(i,3)>0&&pref(i+1,3)==0 %左脚离地时刻
            if i+1+doubleN<length(t)
                Lfoot(i+1+doubleN:end,2)=10^10;
            end
        end
        if pref(i,3)==0&&pref(i+1,3)<0 %右脚落地时刻
            if i+1-doubleN<0
                Rfoot(:,2)=pref(i+1,2);
            else
                Rfoot(i+1-doubleN:end,2)=pref(i+1,2);
            end
        end
        if pref(i,3)<0&&pref(i+1,3)==0 %右脚离地时刻
            if i+1+doubleN<length(t)
                Rfoot(i+1+doubleN:end,2)=10^10;
            end
        end
    end
    %结束双支撑
    Lfoot(length(t)-round(Tstep/dt)-doubleN:end,2)=pref(end,2);
    Rfoot(length(t)-round(Tstep/dt)-doubleN:end,2)=pref(end,2);
    
    %离地区间插值
    %左腿
    while 1==1
        a0=[];
        for i=1:length(t)-1
            if Lfoot(i,2)<10^10-1 && Lfoot(i+1,2)>10^10-1 %左脚离地
                a0=Lfoot(i,2);
                t0=i;
            end
            if Lfoot(i,2)>10^10-1 && Lfoot(i+1,2)<10^10-1 %左脚着地
                a1=Lfoot(i+1,2);
                t1=i+1;
            end
        end
        if isempty(a0)
            break
        else
            [q,qd,qdd]= fun_quintic_polynomial_interpolation(a0,a1,0,0,0,0,t1-t0,(t0:t1)-t0);
            Lfoot(t0:t1,2)=q;
            [q1,qd,qdd]= fun_quintic_polynomial_interpolation(0,H,0,0,0,0,round((t1-t0)/2),(t0:t0+round((t1-t0)/2))-t0);
            [q2,qd,qdd]= fun_quintic_polynomial_interpolation(H,0,0,0,0,0,t1-(t0+round((t1-t0)/2)+1),(t0+round((t1-t0)/2)+1:t1)-(t0+round((t1-t0)/2)+1));
            Lfoot(t0:t1,4)=[q1,q2];
        end
    end
    %右腿
    while 1==1
        a0=[];
        for i=1:length(t)-1
            if Rfoot(i,2)<10^10-1 && Rfoot(i+1,2)>10^10-1 %左脚离地
                a0=Rfoot(i,2);
                t0=i;
            end
            if Rfoot(i,2)>10^10-1 && Rfoot(i+1,2)<10^10-1 %左脚着地
                a1=Rfoot(i+1,2);
                t1=i+1;
            end
        end
        if isempty(a0)
            break
        else
            [q,qd,qdd]= fun_quintic_polynomial_interpolation(a0,a1,0,0,0,0,t1-t0,(t0:t1)-t0);
            Rfoot(t0:t1,2)=q;
            [q1,qd,qdd]= fun_quintic_polynomial_interpolation(0,H,0,0,0,0,round((t1-t0)/2),(t0:t0+round((t1-t0)/2))-t0);
            [q2,qd,qdd]= fun_quintic_polynomial_interpolation(H,0,0,0,0,0,t1-(t0+round((t1-t0)/2)+1),(t0+round((t1-t0)/2)+1:t1)-(t0+round((t1-t0)/2)+1));
            Rfoot(t0:t1,4)=[q1,q2];
        end
    end
    
    gait=[t',gaitx(:,4),gaity(:,4),Lfoot(:,2:4),Rfoot(:,2:4)];