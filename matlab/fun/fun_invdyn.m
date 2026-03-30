function torq=fun_invdyn(Ndof,T,m,c,I,q,qd,qdd)
    %Ndof自由度数
    %T表示0位状态下的，各关节相对上一关节的configuration，用齐次变换矩阵表示，共4*Ndof行4列
    %m,c,I表示关节质量、质心（相对关节坐标系，3行Ndof列）、惯性张量（相对质心，对齐关节坐标系，3*Ndof行3列）
    %q,qd,qdd关节角度、速度、加速度
    g=9.8;
    
    %计算关节位置矢量p和关节轴方向矢量a
    p=zeros(3,Ndof);
    a=zeros(3,Ndof);
    R=zeros(3*Ndof,3);
    for i=1:Ndof
        Ti=eye(4);
        for j=1:i
            Ti=Ti*T((4*j-3):(4*j),:)*[cos(q(j)),-sin(q(j)),0,0;sin(q(j)),cos(q(j)),0,0;0,0,1,0;0,0,0,1];
        end
        p(:,i)=Ti(1:3,4);
        a(:,i)=Ti(1:3,1:3)*[0,0,1]';
        R(3*i-2:3*i,:)=Ti(1:3,1:3);
    end
    
    %计算空间速度及其微分
    s=zeros(6,Ndof);  %关节轴位置和方向组成的矢量
    for i=1:Ndof
        s(:,i)=[cross(p(:,i),a(:,i));a(:,i)];
    end
    
    spatial_velocity=zeros(6,Ndof);
    spatial_velocity(:,1)=qd(1)*s(:,1);
    for i=2:Ndof
        spatial_velocity(:,i)=spatial_velocity(:,i-1)+qd(i)*s(:,i);
    end
    
    sd=zeros(6,Ndof); %s的一阶导数
    for i=2:Ndof
        omegai_m1_hat=cross_matrix(spatial_velocity(4:6,i-1));
        v_o_m1_hat=cross_matrix(spatial_velocity(1:3,i-1));
        sd(:,i)=[omegai_m1_hat,v_o_m1_hat;zeros(3,3),omegai_m1_hat]*s(:,i);
    end
    
    spatial_velocity_dot=zeros(6,Ndof);
    spatial_velocity_dot(:,1)=s(:,1)*qdd(1);
    for i=2:Ndof
        spatial_velocity_dot(:,i)=spatial_velocity_dot(:,i-1)+sd(:,i)*qd(i)+s(:,i)*qdd(i);
    end
    
    % 构造关节惯量矩阵
    Is=zeros(6*Ndof,6);
    for i=1:Ndof
        ci=p(:,i)+R(3*i-2:3*i,:)*c(:,i);
        ci_hat=cross_matrix(ci);
        Ii=R(3*i-2:3*i,:)*I(3*i-2:3*i,:)*R(3*i-2:3*i,:)';
        Is(6*i-5:6*i,:)=[m(i)*eye(3),m(i)*ci_hat';
                         m(i)*ci_hat,m(i)*(ci_hat*ci_hat')+Ii];
    end
    
    % 重力
    fe=zeros(6,Ndof);
    for i=1:Ndof
        G=[0,0,-m(i)*g]';
        ci=p(:,i)+R(3*i-2:3*i,:)*c(:,i);
        fe(:,i)=[G;cross(ci,G)];
    end
    
    %牛顿欧拉方程求解
    f=zeros(6,Ndof);
    Isi=Is(6*Ndof-5:6*Ndof,:);
    svdi=spatial_velocity_dot(:,Ndof);
    svi=spatial_velocity(:,Ndof);
    omega_i_hat=cross_matrix(svi(4:6));
    v0_i_hat=cross_matrix(svi(1:3));
    svi_cross=[omega_i_hat,zeros(3,3);v0_i_hat,omega_i_hat];
    f(:,Ndof)=Isi*svdi+svi_cross*Isi*svi-fe(:,Ndof);
    for i=Ndof-1:-1:1
        Isi=Is(6*i-5:6*i,:);
        svdi=spatial_velocity_dot(:,i);
        svi=spatial_velocity(:,i);
        omega_i_hat=cross_matrix(svi(4:6));
        v0_i_hat=cross_matrix(svi(1:3));
        svi_cross=[omega_i_hat,zeros(3,3);v0_i_hat,omega_i_hat];
        G=[0,0,-m(i)*g]';
        f(:,i)=Isi*svdi+svi_cross*Isi*svi-fe(:,i)+f(:,i+1);
    end
    
    %计算关节力矩
    torq=zeros(1,Ndof);
    for i=1:Ndof
        torq(i)=s(:,i)'*f(:,i);
    end
    
    end
    
    function a=cross_matrix(b)
    a=[0,-b(3),b(2);
       b(3),0,-b(1);
       -b(2),b(1),0];
    end