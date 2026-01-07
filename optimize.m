function [Mdl,fMin,Convergence_curve]  = optimize_fitrMLP(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti)  
     pop=num_pop;
     M=num_iter;
     LB=[1,2,2,0.001];
     UB=[3,25,25,0.1];
     nvars=length(LB);
     fit_fitrensemble1=@fit_fitrMLP;
    if strcmp( method_mti,'PSO粒子群算法')==1 
      [fMin,Mdl,Convergence_curve,pos]=PSO_PSO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);         
    else
        error('Unsupported method_mti in this minimal optimizer.');
    end
    if(pos(1)<=1)
        hid_size=round(pos(2));
    else
        hid_size=[round(pos(2)),round(pos(3))];  
    end
     figure
     plot(Convergence_curve,'--p','LineWidth',1.2,'Color',[160,123,194]./255) 
     xticks([1:length(Convergence_curve)])
     title('optimize process')
     xlabel('iter')
     ylabel('fitness')
     grid on
   set(gca,"FontName","Times New Roman","FontSize",12,"LineWidth",1.2)
   box off 
 disp([method_mti, '优化 fitrnet:   ',"LayerSizes:",num2str(hid_size),'   Lambda: ',num2str((pos(4)))]) 
end

function [fitness_value,Mdl]=fit_fitrMLP(pop,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%    global train_x_feature_label_norm train_y_feature_label_norm vaild_x_feature_label_norm vaild_y_feature_label_norm
if(pop(1)<=2)
    Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",round(pop(2)),"Lambda",(pop(4))); %神经元和激活函数,'KernelScale',pop(1),'Epsilon',pop(2));  
else
   Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",round([pop(2),pop(3)]),"Lambda",(pop(4))); %神经元和激活函数,'KernelScale',pop(1),'Epsilon',pop(2));  
% elseif (pop(1)==3)
%   Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",[pop(2),pop(3),pop(4)]); %神经元和激活函数,'KernelScale',pop(1),'Epsilon',pop(2));  
end
    P_vaild_y_feature_label_norm=predict(Mdl,vaild_x_feature_label_norm);
    fitness_value=sum(abs(P_vaild_y_feature_label_norm-vaild_y_feature_label_norm))/length(vaild_y_feature_label_norm);
end

function [gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%PSO Infotmation
% if(max(size(ub)) == 1)
    UB=ub;
    LB=lb;
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
% end

Vmax=6;
noP=N;
wMax=0.9;
wMin=0.6;
c1=2;
c2=2;

% Initializations
iter=Max_iteration;
vel=zeros(noP,dim);
pBestScore=zeros(noP);
pBest=zeros(noP,dim);
gBest=zeros(1,dim);
cg_curve=zeros(1,iter);
vel=zeros(N,dim);
pos=zeros(N,dim);

%Initialization
for i=1:size(pos,1) 
    for j=1:size(pos,2) 
        pos(i,j)=(ub(j)-lb(j))*rand()+lb(j);
        vel(i,j)=0.3*rand();
    end
end

for i=1:noP
    pBestScore(i)=inf;
end

% Initialize gBestScore for a minimization problem
 gBestScore=inf;
     
 [~,Mdl]=fobj(pos(1,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
  Mdl_best=Mdl; 
for l=1:iter 
    
    % Return back the particles that go beyond the boundaries of the search
    % space
     Flag4ub=pos(i,:)>ub;
     Flag4lb=pos(i,:)<lb;
     pos(i,:)=(pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
%      disp(pos)
    for i=1:size(pos,1)     
        %Calculate objective function for each particle
        [fitness,Mdl]=fobj(pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pBest(i,:)=pos(i,:);
            Mdl_best=Mdl;
        end
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=pos(i,:);
        end
    end

    %Update the W of PSO
    w=wMax-l*((wMax-wMin)/iter);
    %Update the Velocity and Position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)       
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));
            
            if(vel(i,j)>Vmax)
                vel(i,j)=Vmax;
            end
            if(vel(i,j)<-Vmax)
                vel(i,j)=-Vmax;
            end            
            pos(i,j)=pos(i,j)+vel(i,j);
            if (pos(i,j)>UB(j))
                pos(i,j)=UB(j);
            elseif (pos(i,j)<LB(j))
                pos(i,j)=LB(j);
            end
        end
    end
    cg_curve(l)=gBestScore;
end

end
%% 模拟退火
%[gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,data_O,data_Z)

function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
%%
% Application of simple limits/bounds

function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
  s(1:3)=round(s(1:3));
end
