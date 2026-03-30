function pop = PSO_MO_main(p_range,p_discrete,Ini_gene_method,p0_ini_para,NO)
    %多目标粒子群算法主函数
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %输入参数：
    %p_range,参数取值范围，用于随机产生初始种群，2行n列，第一行是参数下限，第二行是参数上限，
    %p_discrete,共n项，表示设计变量是连续的（0），只能取整（1）,或指定取值精度（n>0）
    %Ini_gene_method初始种群的定义方式，0表示完全随机产生，1表示给定满足限制条件的初始种群，2计算中断后可继续计算（需要有输出文件pop.mat）；
    %p0_ini_para定义初值所需参数，初值定义方法为0或2时，本参数无意义，初值方法为1时，本参数为初始种群数据，共size_pop行n+NO列，需满足限制条件，前n列为设计变量值，后NO列为目标函数值；
    %NO优化目标的个数；
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %输出参数：
    %pop优化过程详细信息，包含每一代种群的：
    % 设计变量（size_pop行n列）、移动速度（size_pop行n列）、目标函数值（size_pop行NO列），
    % 群体非劣解集（x行n列），群体非劣解的目标函数值（x行NO列），群体最优解（1*n），群体最优解目标函数值（1*NO）
    % 个体最优解（size_pop行n列）、个体最优解目标函数值（size_pop行NO列）
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %需定义以下子函数，以供调用
    %限制条件J_lim=estimate_limit(p,p_range);满足限制条件返回1，否则返回0
    %目标函数Obj=estimate_Obj(p);计算目标函数值，返回目标值1*NO，以目标值更小为优
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %关于限制条件的处理方式
    %1——如不满足限制条件，重新移动，直至满足，为防止陷入局部死循环，设定不满足的次数越多，发生变异的概率越高，此种策略适用于限制条件较强的情况
    %2——如不满足限制条件，目标函数取一极大值，以便完全阻止其成为非劣解，此种策略适用于限制条件较弱，即很容易满足的情况
    limit_method=1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %自适应网格尺寸：自适应网格技术是为了实现种群的多样性，将非劣解集进行网格划分，选择密度小的作为全局最优。同时删减非劣解集
    grid_num=20; %对非劣解集进行网格划分的数量
    noninf_num=200; %限定非劣解集的数量上限
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %算法参数定义
    %种群数量
    size_pop=100;
    %最大代数
    max_gen=200;
    %加速度因子,一般取0-4
    c1=1.495;
    c2=1.495;
    %初始惯性权重和最终惯性权重
    w_start=0.8;
    w_end=0.2;
    %种群总变异概率
    km=0.99/size_pop;
    
    
    %变量维度
    n=length(p_range(1,:));
    %初始化输出参数，以结构体的形式储存优化过程信息。
    pop=struct;
    
    %定义速度最大值和最小值：参数一般取取值范围的（0.05-0.5）
    V_max=(p_range(2,:)-p_range(1,:))*0.5;
    V_min=-V_max;
    
    %初始化种群
    if Ini_gene_method==0
        p=zeros(size_pop,n);  %参数
        V=zeros(size_pop,n);  %速度
        f=zeros(size_pop,NO); %适应度
        for i=1:size_pop
            while 1==1
                p(i,:)=rand(1,n).*(p_range(2,:)-p_range(1,:))+p_range(1,:); %随机产生初值
                p(i,:)=get_discrete(p(i,:),p_discrete);
                lim=estimate_limit(p(i,:),p_range);
                if lim==1 %满足限制条件
                    f(i,:)=estimate_Obj(p(i,:));
                    break
                end
            end
            %V(i,:)=rand(1,n).*(V_max-V_min)+V_min; %可以随机定义初始速度，也可以定义为0
            gen_ini_pop=i   %实时显示初始化进度
        end
    elseif Ini_gene_method==1
        p=p0_ini_para(:,1:n);
        f=p0_ini_para(:,n+1:n+NO);
        V=zeros(size_pop,n);
    %   for i=1:size_pop
    %       V(i,:)=rand(1,n).*(V_max-V_min)+V_min;   %可以随机定义初始速度，也可以定义为0
    %   end
    elseif Ini_gene_method==2
        %不产生初始种群，读取种群数据继续计算
    end
    
    %根据初始种群，计算非劣解集，种群最优解和个体最优解
    if Ini_gene_method~=2
        %获得初始非劣解集
        [pnonp,pnonf]=get_noninferior(p,f);
        %初始极值和适应度
        ibestp=p; %个体极值
        ibestf=f; %个体极值适应度
        %群体极值按照网络密度进行选择
        grid_inform=grid_noninferior(pnonf,grid_num);
        [pbestp,pbestf]=select_pbest(pnonp,pnonf,grid_inform,size_pop);
    else
        pop=load('pop.mat','pop'); %读取已有的数据，继续计算
        pop=pop.pop;
        s=length(pop);
        p=pop(s).pop;
        V=pop(s).V;
        f=pop(s).fitness;
        pnonp=pop(s).noninferp;
        pnonf=pop(s).noninferf;
        grid_inform=pop(s).grid_inform;
        pbestp=pop(s).pbestpop;
        pbestf=pop(s).pbestfitness;
        ibestp=pop(s).ibestpop;
        ibestf=pop(s).ibestfitness;
    end
    
    
    %开始迭代
    if Ini_gene_method~=2
        t=1;
    else
        t=s;
    end
    
    while t<=max_gen
        %记录当前代的结果并实时保存
        pop(t).pop=p;
        pop(t).V=V;
        pop(t).fitness=f;
        pop(t).noninferp=pnonp;
        pop(t).noninferf=pnonf;
        pop(t).pbestpop=pbestp;
        pop(t).pbestfitness=pbestf;
        pop(t).ibestpop=ibestp;
        pop(t).ibestfitness=ibestf;
        pop(t).grid_inform=grid_inform;
        save pop
        
        %画出当前代的非劣解集，暂只适用于2个或3个目标
        if NO==2
            clf
            scatter(pnonf(:,1),pnonf(:,2));
            pause(0.01)   %短暂暂停以保证图成功显示
        elseif NO==3
            clf
            scatter3(pnonf(:,1),pnonf(:,2),pnonf(:,3));
            pause(0.01)   %短暂暂停以保证图成功显示
        end
            
        
        %惯性权重，可以选择如下几种形式
        %w=0.5;
        w=w_start-(w_start-w_end)*(t/max_gen);
        %w=w_start-(w_start-w_end)*(t/max_gen)^2;
        %w=w_start-(w_start-w_end)*(2*t/max_gen-(t/max_gen)^2);
        %w=w_end*(w_start/w_end)^(1/(1+10*t/max_gen));
        
        
        %更新粒子种群的位置
        
        if limit_method==1 
            %%限制条件策略：如不满足限制条件，重新产生移动，直至满足，为防止陷入局部死循环，设定不满足的次数越多，发生变异的概率越高，但有上限。
            for i=1:size_pop
                kc=1;
                while 1==1
                    %更新速度
                    V(i,:)=w*V(i,:)+c1*rand.*(ibestp(i,:)-p(i,:))+c2*rand.*(pbestp(i,:)-p(i,:));
                    for j=1:n
                        if V(i,j)>V_max(j)
                            V(i,j)=V_max(j);
                        elseif V(i,j)<V_min(j)
                            V(i,j)=V_min(j);
                        end
                    end
                    
                    %更新位置
                    p(i,:)=p(i,:)+V(i,:);
                    p(i,:)=get_discrete(p(i,:),p_discrete);
                    
                    %自适应变异，如果多次循环均未获得满足限制条件的粒子位置，增加变异率
                    if rand<min(km*kc/10,0.1)
                        p(i,:)=rand(1,n).*(p_range(2,:)-p_range(1,:))+p_range(1,:);
                        p(i,:)=get_discrete(p(i,:),p_discrete);
                    end
    
                    %判断是否满足限制条件，如不满足则重新移动
                    J_lim=estimate_limit(p(i,:),p_range);
                    if J_lim==1
                        Obj=estimate_Obj(p(i,:));
                        f(i,:)=Obj;
                        break
                    end
                    kc=kc+1;
                end
                
                step1=t+i/1000  %实时显示当前计算进度
            end
            
        elseif limit_method==2
            
            %限制条件策略：如不满足限制条件，目标函数取极大值，以便完全阻止其成为非劣解。
            for i=1:size_pop
                %更新速度
                V(i,:)=w*V(i,:)+c1*rand.*(ibestp(i,:)-p(i,:))+c2*rand.*(pbestp(i,:)-p(i,:));
                for j=1:n
                    if V(i,j)>V_max(j)
                        V(i,j)=V_max(j);
                    elseif V(i,j)<V_min(j)
                        V(i,j)=V_min(j);
                    end
                end
                
                %更新位置
                p(i,:)=p(i,:)+V(i,:);
                p(i,:)=get_discrete(p(i,:),p_discrete);
                
                %参数上下限修正
                for j=1:n
                   if p(i,j)>p_range(2,j)
                       p(i,j)=p_range(2,j);
                   elseif p(i,j)<p_range(1,j)
                       p(i,j)=p_range(1,j);
                   end
                end
                
                %自适应变异，如果多次循环均未获得满足限制条件的粒子位置，增加变异率
                if rand<km
                    p(i,:)=rand(1,n).*(p_range(2,:)-p_range(1,:))+p_range(1,:);
                    p(i,:)=get_discrete(p(i,:),p_discrete);
                end
    
                %判断是否满足限制条件，如满足，计算目标函数，否则目标函数取极大值
                J_lim=estimate_limit(p(i,:),p_range);
                if J_lim==1
                    f(i,:)=estimate_Obj(p(i,:));
                else
                    f(i,:)=ones(1,NO)*10^10;
                end
                step=t+i/1000
            end
        end
                       
        %更新个体极值
        for i=1:size_pop
            if sum(f(i,:)<ibestf(i,:))==NO
                ibestf(i,:)=f(i,:);
                ibestp(i,:)=p(i,:);
            elseif sum(f(i,:)<ibestf(i,:))>0&&sum(f(i,:)<ibestf(i,:))<NO
                if rand>=0.5
                    ibestf(i,:)=f(i,:);
                    ibestp(i,:)=p(i,:);
                end
            end
        end
        
        %更新群体非劣解集
        %首先求当前代的非劣解集
        [pnonp_new,pnonf_new]=get_noninferior(p,f);
        %将历史非劣解集和当前代非劣解集合并，求出新的非劣解集
        pnonp_new_candidate=[pnonp_new;pnonp];
        pnonf_new_candidate=[pnonf_new;pnonf];
        [pnonp,pnonf]=get_noninferior(pnonp_new_candidate,pnonf_new_candidate);
        %非劣解集网格划分
        grid_inform=grid_noninferior(pnonf,grid_num);
        %非劣解集删减
        [pnonp,pnonf]=cut_noninferior(pnonp,pnonf,grid_inform,noninf_num);
        %重新网格划分
        grid_inform=grid_noninferior(pnonf,grid_num);
        %更新群体极值和群体适应度
        [pbestp,pbestf]=select_pbest(pnonp,pnonf,grid_inform,size_pop);
    
        t=t+1;
    end
    
end
    
    
function [pnonp,pnonf]=get_noninferior(p,f)
    %由解集p及其目标值f，求出解集的非劣解集及目标值，并划分网格
    %判断解之间是否支配，第(i,j)项表示第i个解是否被第j个解支配，0表示不支配，1表示支配
    NO=length(f(1,:));
    np=length(p(:,1));
    dom=zeros(np,np);
    for i=1:(np-1)
        for j=(i+1):np
            ifdom=sum(f(i,:)<=f(j,:));
            if ifdom==0 %i被j支配
                dom(i,j)=1;
                dom(j,i)=0;
            elseif ifdom==NO %j被i支配
                dom(i,j)=0;
                dom(j,i)=1;
            else %互不支配
                dom(i,j)=0;
                dom(j,i)=0;
            end
        end
    end
    %非劣解集
    sdom=sum(dom,2);
    nond=find(sdom==0);
    pnonp=p(nond,:);
    pnonf=f(nond,:);
end
    
function grid_inform=grid_noninferior(pnonf,grid_num)
    %对非劣解集划分网格，返回网格密度信息
    %以结构体形式储存密度信息，包括有grid（所属网格），density(点的数量)，non_num(属于该网格的点的编号，对应pnonp的行数)

    %非劣解集的size
    Nnon=length(pnonf(:,1));
    NO=length(pnonf(1,:));
    %网格的范围
    grid_limit=zeros(2,NO);
    grid_size=zeros(1,NO);
    for i=1:NO
        maxf=max(pnonf(:,i));
        minf=min(pnonf(:,i));
        grid_size(i)=(maxf-minf)/(grid_num-1);
        grid_limit(1,i)=minf-grid_size(i)/2;
        grid_limit(2,i)=maxf+grid_size(i)/2;
    end
    %所属网格
    dgrid=[]; %用于保存密度大于1的网格的号码
    grid_inform=struct;%存储网格密度和属于该网格的非劣解的序号
    for i=1:Nnon
        gridi=zeros(1,NO);
        for j=1:NO
            gridi(j)=ceil((pnonf(i,j)-grid_limit(1,j))/grid_size(j));
        end
        s=size(dgrid);
        k=1;
        while k<=s(1)
            if sum(gridi==dgrid(k,:))==NO
                grid_inform(k).grid=gridi;
                grid_inform(k).density=grid_inform(k).density+1;
                grid_inform(k).non_num=[grid_inform(k).non_num,i];
                break
            end
            k=k+1;
        end
        if k>s(1)
            dgrid=[dgrid;gridi];
            grid_inform(k).grid=gridi;
            grid_inform(s(1)+1).density=1;
            grid_inform(s(1)+1).non_num=i;
        end
    end
end
    
function [pnonp,pnonf]=cut_noninferior(pnonp,pnonf,grid_inform,noninf_num)
    %对非劣解集进行删减
    Nnon=length(pnonf(:,1));
    if Nnon>noninf_num
        Ngrid=length(grid_inform);
        Nun_non=[];
        for i=1:Ngrid
            delN=round((Nnon-noninf_num)/Nnon*grid_inform(i).density);
            if delN<grid_inform(i).density*0.8
                nonnum=grid_inform(i).non_num;
                for j=1:delN
                    dN=ceil(rand*(grid_inform(i).density+1-j));
                    nonnum(dN)=[];
                end
                grid_inform(i).non_num=nonnum;
            end
            Nun_non=[Nun_non,grid_inform(i).non_num];
        end
        pnonp=pnonp(Nun_non,:);
        pnonf=pnonf(Nun_non,:);
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
    
function [pbestp,pbestf]=select_pbest(pnonp,pnonf,grid_inform,size_pop)
    %采用网格法选择群体极值
    Ngrid=length(grid_inform);
    dens=zeros(1,Ngrid);
    pp=zeros(1,Ngrid+1);
    for i=1:Ngrid
        dens(i)=grid_inform(i).density;
    end
    for i=1:Ngrid
        pp(i+1)=sum(1./dens(1:i))/sum(1./dens);
    end
    
    n=length(pnonp(1,:));
    NO=length(pnonf(1,:));
    pbestp=zeros(size_pop,n);
    pbestf=zeros(size_pop,NO);
    for i=1:size_pop
        kk=rand;
        for j=1:Ngrid
            if kk>=pp(j)&&kk<=pp(j+1)
                idgrid=j;
                break
            end
        end
        idp=ceil(rand*grid_inform(idgrid).density);
        gridp=grid_inform(idgrid).non_num;
        sele_n=gridp(idp);
        pbestp(i,:)=pnonp(sele_n,:);
        pbestf(i,:)=pnonf(sele_n,:);
    end
end