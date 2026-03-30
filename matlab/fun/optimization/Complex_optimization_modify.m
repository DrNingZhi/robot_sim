function pop = Complex_optimization_modify(p_range,p_discrete,size_pop,Ini_gene_method,p0_ini_para,Conv_method,Obj_conv)
    %复合形法进行参数优化，输出最优解（目标函数最小化），和解集迭代过程，目标函数迭代过程
    
    %输入参数说明
    %1.参数取值范围p_range，两行n列，每列表示一个参数的上下限,第一行表示下限，第二行表示上限。
    %2.参数的连续性p_discrete,共n项，表示设计变量是连续的（0），只能取整（1）,或指定取值精度（n>0）。
    %3.种群数量P，建议取多于 (参数个数*10)。
    %4.初始解集生成方法Ini_gene_method，0表示完全随机生成，1表示完全给定初始解集,2表示继续计算（需要中断前的数据）。
    %5.初始解集生成参数p0_ini_para，方法为0或2时无用，1时为初始单解以及随机范围。
    %6.判断收敛方式Conv_method，0表示目标函数值偏差决定,Obj_conv表示收敛系数，1表示解集偏差决定，Obj_conv表示每个解的收敛系数.
    
    %调用函数说明
    %需要编写子函数[J_lim,Obj]=estimate_Obj_limit(p,p_range)判断限制条件和计算目标函数
    %输入参数取值和参数范围
    %输出J_lim=1表示满足限制条件，输出J_lim=0表示不满足限制条件
    %输出Obj表示最小化目标函数值（单目标）
    
    %输出参数说明
    %pop：结构体，包含：每步的解集p、目标函数值Obj、最优解bestp、最优解目标函数值bestObj，无需等待计算完成，每步迭代完成后均会保存当前数据
    
    pop=struct;
    
    %参数维度
    n=length(p_range(1,:));
    
    %定义优化参数
    alpha=1.3; %反射系数
    
    %生成初始解集
    p0=zeros(size_pop,n); %每行表示一个解
    Obj0=zeros(size_pop,1); %目标函数值
    if Ini_gene_method==0
        %完全随机法
        for i=1:size_pop
            while 1==1 %产生满足限制条件的初始解
                p0(i,:)=rand(1,n).*(p_range(2,:)-p_range(1,:))+p_range(1,:);
                p0(i,:)=get_discrete(p0(i,:),p_discrete);
                [J_lin,Obj]=estimate_Obj_limit(p0(i,:),p_range);
                if J_lin==1 %判断是否满足限制条件,均满足返回1，有不满足返回0
                    Obj0(i)=Obj; %目标函数值
                    break
                end
            end
            initial_generation=i
        end
    elseif Ini_gene_method==1
        %给定完整初值
        p0=p0_ini_para(:,1:n);
        Obj0=p0_ini_para(:,n+1);
    elseif Ini_gene_method==2
        %继续计算
        aa=load('pop.mat');
        pop=aa.pop;
        s=length(pop);
        p0=pop(s).p;
        Obj0=pop(s).Obj;
    elseif Ini_gene_method==5
        %给定部分初值
        ni=length(p0_ini_para(:,1));
        p0(1:ni,:)=p0_ini_para(:,1:n);
        Obj0(1:ni)=p0_ini_para(:,n+1);
        for i=(ni+1):size_pop
            while 1==1 %产生满足限制条件的初始解
                p0(i,:)=rand(1,n).*(p_range(2,:)-p_range(1,:))+p_range(1,:);
                p0(i,:)=get_discrete(p0(i,:),p_discrete);
                [J_lin,Obj]=estimate_Obj_limit(p0(i,:),p_range);
                if J_lin==1 %判断是否满足限制条件,均满足返回1，有不满足返回0
                    Obj0(i)=Obj; %目标函数值
                    break
                end
            end
            initial_generation=i
        end
    end
    
    
    
    p=p0;
    Obj_value_1=Obj0;
    
    %优化迭代
    if Ini_gene_method==2 %继续优化
        step=s;
    else %从头优化
        step=1;
    end
    
    
    while 1==1
        %记录当前步结果
        pop(step).p=p;
        pop(step).Obj=Obj_value_1;
        [minp,mind]=min(Obj_value_1);
        pop(step).bestObj=minp;
        pop(step).bestp=p(mind,:);
        save pop;
        
        %画出收敛过程
        Obj_value=zeros(size_pop,step);
        t=1:step;
        for i=1:step
            Obj_value(:,i)=pop(i).Obj;
        end
        plot(t,Obj_value);
        pause(0.01)
        
        %收敛判断
        [maxp,maxd]=max(Obj_value_1);
        pw=p(maxd,:);
        pb=p(mind,:);
        if Conv_method==0
            %如果最优和最差解的目标函数值接近，则收敛，优化结束
            if abs(max(Obj_value_1)-min(Obj_value_1))<Obj_conv
                break
            end
        else
            %如果最优和最差解接近，则收敛，优化结束
            conv=(abs(pb-pw)<Obj_conv);
            if sum(conv)==n
                break
            end
        end
       
        %优化迭代
        req=1; %最差解次序
        while req<=size_pop/3
            
            %找到最优解，第reg最差解，除最差解外的平均解，
            [sortp,sortd]=sort(Obj_value_1);
            mw=sortd(size_pop-req+1);
            bestp=sortp(size_pop-req+1);
            pw=p(mw,:); %第req差的解
            p_dw=p;
            p_dw(mw,:)=[]; 
            pc=mean(p_dw); %去掉最差解后的平均解
            
            %更新最差解
            yn=0; %迭代失败次数
            while yn<50 
                if yn==0
                    pcand=pc+alpha*(pc-pw);
                else
                    beta=1+(yn-1)/4;
                    ep=beta^(-beta);
                    pcand=0.5*(pcand+ep*pc+(1-ep)*pb)+(pc-pb)*(1-ep)*(2*rand(1,1)-1);
                end
                pcand=get_discrete(pcand,p_discrete);
                %判断是否满足限制条件并且更优
                [J_lin,Obj]=estimate_Obj_limit(pcand,p_range);
                if J_lin==1&&Obj<bestp
                    Obj_re=Obj;
                    break
                else
                    yn=yn+1;
                end
            end
            
            %如果当前最差解多次迭代失败，更换次差解
            if yn<50
                p(mw,:)=pcand;
                Obj_value_1(mw)=Obj_re;
                break
            else
                req=req+1;
            end
        end
        
        if req>size_pop/3
            disp('死循环')
            break
        end
        
        step=step+1
    end
    
end
    
    function p=get_discrete(p,p_discrete)
        %按照离散度定义，更新参数
        n=length(p);
        for j=1:n
            if p_discrete(j)==0
                p(j)=p(j);
            elseif p_discrete(j)==1
                p(j)=round(p(j));  %取整
            else
                dis=mod(p(j),p_discrete(j));
                if dis/p_discrete(j)<0.5
                    p(j)=p(j)-dis;
                else
                    p(j)=p(j)+p_discrete(j)-dis;
                end
            end
        end
    end
    