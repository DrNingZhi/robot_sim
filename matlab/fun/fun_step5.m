function p = fun_step5(t,t0,p0,t1,p1,order)
    %五次多项式插值，表示：
    %当t<t0，p=p0;
    %当t>t1，p=p1;
    %当t0<t<t1，五次多项式插值
    if t<t0
        if order==0
            p=p0;
        else
            p=0;
        end
    elseif t>t1
        if order==0
            p=p1;
        else
            p=0;
        end
    else
        A=[t0^5,t0^4,t0^3,t0^2,t0,1;
           5*t0^4,4*t0^3,3*t0^2,2*t0,1,0;
           20*t0^3,12*t0^2,6*t0,2,0,0;
           t1^5,t1^4,t1^3,t1^2,t1,1;
           5*t1^4,4*t1^3,3*t1^2,2*t1,1,0;
           20*t1^3,12*t1^2,6*t1,2,0,0];
        B=[p0,0,0,p1,0,0]';
        a=pinv(A)*B;
        if order==0
            p=a(1)*t^5+a(2)*t^4+a(3)*t^3+a(4)*t^2+a(5)*t+a(6);
        elseif order==1
            p=5*a(1)*t^4+4*a(2)*t^3+3*a(3)*t^2+2*a(4)*t+a(5);
        else
            p=20*a(1)*t^3+12*a(2)*t^2+6*a(3)*t+2*a(4);
        end
    end
    end