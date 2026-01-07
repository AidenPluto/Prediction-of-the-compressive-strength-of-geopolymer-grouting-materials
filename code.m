clc;clear;close all;	
load('R_17_Apr_2025_15_43_14.mat')	
random_seed=1;   	
rng(random_seed)  	
	
data_str='C:\Users\UserX\Desktop\Train.xlsx';   	
	
	
data1=readtable(data_str,'VariableNamingRule','preserve');  	
data2=data1(:,2:end); 	
data=table2array(data1(:,2:end));	
data_biao=data2.Properties.VariableNames;  	
str_label=0; 	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	

x_feature_label=data_select(:,1:end-1);    	
y_feature_label=data_select(:,end);          	
index_label1=1:(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=[7,1.5,1.5];  	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); 	
 	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;      	
	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  	
	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;     	
	

num_pop=8;   	
num_iter=8;   	
method_mti='PSO粒子群算法';   	
min_batchsize=58;   	
max_epoch=100;   	
hidden_size=[32,32];   	
attention_label=1;   	
attention_head=[1,2];   		
	
disp('MLP回归')	
t1=clock; 	
hidden_size=[32,32];    	
	
[Mdl,fitness,Convergence_curve] = optimize_fitrMLP(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti);    	
y_train_predict_norm=predict(Mdl,train_x_feature_label_norm);  
y_vaild_predict_norm=predict(Mdl,vaild_x_feature_label_norm);  	
y_test_predict_norm=predict(Mdl,test_x_feature_label_norm);  	
t2=clock;	
Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
	
	
	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;   	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 	
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
 train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
 train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
 train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
 train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集R方系数R2：',num2str(train_R2)]) 	
 vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
 vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
 vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
 vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
 vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集R方系数R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
 test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
 test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集R方系数R2：',num2str(test_R2)]) 	
 disp(['算法运行时间Time: ',num2str(Time)])	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    	
y_feature_label_norm_all=(y_feature_label-y_mu)/y_sig;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); 
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
   Mdlkf=fitrnet(x_feature_label_norm_all_traink,y_feature_label_norm_all_traink,'LayerSizes',Mdl.LayerSizes,'Lambda',Mdl.ModelParameters.Lambda);	
	
   Mdl_kfold{1,k}=Mdlkf;	
   y_test_predict_norm_all_testk=predict(Mdlkf,x_feature_label_norm_all_testk); 
   y_test_predict_all_testk=y_test_predict_norm_all_testk*y_sig+y_mu;	
   y_feature_label_all_testk=y_feature_label_norm_all_testk*y_sig+y_mu;	
   test_kfold=sum(abs(y_test_predict_all_testk-y_feature_label_all_testk))/length(y_feature_label_all_testk);
   MAE_kfold(k)=test_kfold;	
	
	
	
end	
	
	
	
	
figure('color',[1 1 1]);	
	
color_set=[0.4902    0.7686    0.6510];	
plot(1:length(MAE_kfold),MAE_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.9*min(MAE_kfold),1.3*max(MAE_kfold)])	
xlabel('kfoldnum')	
ylabel('MAE')	
xticks(1:length(MAE_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(MAE_kfold),'--')	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); 
HBAR = bar(1:length(MAE_kfold),MAE_kfold,1,'EdgeColor','k');	
HBAR(1).FaceColor = color_set;	
xticks(1:length(MAE_kfold))	
xlabel('kfoldnum')	
ylabel('MAE')	
disp('****************************************************************************************')	
disp([num2str(Kfold_num),'折验证预测MAE平均绝对误差结果：'])  	
disp(MAE_kfold)  	
disp([num2str(Kfold_num),'折验证  ','MAE均值为： ' ,num2str(mean(MAE_kfold)),'     MAE标准差为： ' ,num2str(std(MAE_kfold))])  	
	

color_list=G_out_data.color_list;   
rand_list1=G_out_data.rand_list1;   
Line_Width=G_out_data.Line_Width;   
makesize=G_out_data.makesize;   	
yang_str2=G_out_data.yang_str2;   
yang_str3=G_out_data.yang_str3;   	
kuang_width=G_out_data.kuang_width;   
show_num=G_out_data.show_num;   
show_num1=G_out_data.show_num1;   	
show_num2=G_out_data.show_num2;   	
	
FontSize=G_out_data.FontSize;  	
xlabel1=G_out_data.xlabel1;   
ylabel1=G_out_data.ylabel1;   
title1=G_out_data.title1;  	
legend1=G_out_data.legend1;   
box1=G_out_data.box1;  
le_kuang=G_out_data.le_kuang;   	
grid1=G_out_data.grid1;   	
yang_fu3_ku=G_out_data.yang_fu3_ku;  
color_index=G_out_data.color_index;	
yangsi_idnex=G_out_data.yangsi_idnex;  	
  	
  	
	
XX=1:length(train_y_feature_label);	
index_show=1:show_num2;	
	
figure_density(yangsi_idnex(5),y_train_predict(index_show),train_y_feature_label(index_show),'训练集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),train_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
	
plot(gca, XX(index_show),y_train_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'训练集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) 
grid(gca,grid1)	
	
	
	
	
	
XX=1:length(vaild_y_feature_label);	
 index_show=1:show_num1;	
	
	
figure_density(yangsi_idnex(5),y_vaild_predict(index_show),vaild_y_feature_label(index_show),'验证集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),vaild_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
	
plot(gca, XX(index_show),y_vaild_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'验证集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) 
grid(gca,grid1)	
	
	
XX=1:length(test_y_feature_label);	
index_show=1:show_num;	
figure_density(yangsi_idnex(5),y_test_predict(index_show),test_y_feature_label(index_show),'测试集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),test_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
	
plot(gca, XX(index_show),y_test_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'测试集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) 
grid(gca,grid1)	
	
	
	
 
 predict_str=G_out_data.predict_str;   	
	
	
  data1=readtable(predict_str,'VariableNamingRule','preserve');  	
  data_x_pre=table2array(data1(:,2:end));  	
	
	
	
A_data1=data_x_pre; 	
data_select1=A_data1(:,feature_need_last);  	
data_select2=data_select1; 	
test_x_feature_label_norm = (data_select1 - x_mu) ./ x_sig;    
y_test_predict_norm=predict(Mdl,test_x_feature_label_norm);	
	

	
	
y_test_predict_last=y_test_predict_norm*y_sig+y_mu;	
data_str1=cell(length(y_test_predict_last),2);	
if  (str_label==1)	
  for NN=1:length(y_test_predict_last)	
     data_str1{NN,1}=NN;	
     data_str1{NN,2}=data_label_str{y_test_predict_last(NN)};	
  end	
else	
for NN=1:length(y_test_predict_last)	
   for NN2=1:size(y_test_predict_last,2) 	
	
      data_str1{NN,1}=int64(NN);	
	
      data_str1{NN,NN2+1}=y_test_predict_last(NN,NN2);	
    end	
end	
	
end	
	
data_out=cell2table(data_str1);	
disp('data_out')	
%% Supplement: Multi-seed PSO convergence (robustness check)


% 1) 
rng_state_main = rng;

% 2) 
seed_list = 1:20;                
nRuns = numel(seed_list);

% 3) 
all_curve = nan(nRuns, num_iter);
all_bestfitness = nan(nRuns, 1);


% 4) 
for r = 1:nRuns
    rng(seed_list(r), 'twister');    
    
    [Mdl_tmp, fitness_tmp, curve_tmp] = optimize_fitrMLP( ...
        train_x_feature_label_norm, train_y_feature_label_norm, ...
        vaild_x_feature_label_norm, vaild_y_feature_label_norm, ...
        num_pop, num_iter, method_mti);

    all_bestfitness(r) = fitness_tmp;

    
    L = min(numel(curve_tmp), num_iter);
    all_curve(r,1:L) = curve_tmp(1:L);

    
end

% 5) 
rng(rng_state_main);

%% Plot 1 (fixed legend): PSO convergence under multiple random initializations

curves = all_curve;          % size: [nRuns, num_iter]
nRuns  = size(curves,1);
iters  = 1:num_iter;


mean_curve = nanmean(curves,1);
med_curve  = nanmedian(curves,1);
q25 = prctile(curves,25,1);
q75 = prctile(curves,75,1);

figure('Color',[1 1 1], 'Position',[150 150 820 420]); 
hold on; box on;

baseColor = [0.2 0.45 0.85];     

% -------------------------------------------------------------------------
% 1) Individual runs
for r = 1:nRuns
    plot(iters, curves(r,:), '-', ...
        'Color', [baseColor 0.25], ...
        'LineWidth', 1, ...
        'HandleVisibility','off');   
end
h_runs = plot(nan, nan, '-', ...
    'Color', baseColor, 'LineWidth', 1.5);   

% -------------------------------------------------------------------------
% 2) Interquartile range (IQR)
fill([iters fliplr(iters)], [q25 fliplr(q75)], ...
     baseColor, 'FaceAlpha',0.20, ...
     'EdgeColor','none', ...
     'HandleVisibility','off');
h_iqr = patch(nan, nan, baseColor, ...
    'FaceAlpha',0.20, 'EdgeColor','none');

% -------------------------------------------------------------------------
% 3) Median curve
plot(iters, med_curve, '-', ...
     'Color', baseColor, ...
     'LineWidth', 2.4, ...
     'HandleVisibility','off');
h_med = plot(nan, nan, '-', ...
    'Color', baseColor, 'LineWidth', 2.4);

% -------------------------------------------------------------------------
% 4) Mean curve
plot(iters, mean_curve, '--', ...
     'Color', [0 0 0], ...
     'LineWidth', 2.0, ...
     'HandleVisibility','off');
h_mean = plot(nan, nan, '--', ...
    'Color', [0 0 0], 'LineWidth', 2.0);

% -------------------------------------------------------------------------
% Axes & labels
xlabel('Iteration');
ylabel('Best fitness (validation MAE, lower is better)');
title(sprintf('PSO convergence under %d random initializations', nRuns));

ymin = min(curves(:));
ymax = max(curves(:));
pad  = 0.12*(ymax-ymin+eps);
ylim([ymin-pad ymax+pad]);
xlim([1 num_iter]);

grid on;
set(gca,'FontSize',12,'LineWidth',1.2);

% -------------------------------------------------------------------------
% Legend (now 100% correct)
legend([h_runs, h_iqr, h_med, h_mean], ...
       {'Individual runs', ...
        'Interquartile range (25–75%)', ...
        'Median convergence', ...
        'Mean convergence'}, ...
       'Location','northeast');

%% Plot 2 (improved): Final best fitness distribution (boxplot + jitter scatter)
vals = all_bestfitness(:);
nRuns = numel(vals);

mu  = mean(vals);
sg  = std(vals);
mn  = min(vals);
mx  = max(vals);
med = median(vals);
iqr_val = iqr(vals);

figure('Color',[1 1 1], 'Position',[200 200 760 420]); 
hold on; box on;

% 1) 
h = boxplot(vals, 'Widths', 0.45, 'Symbol', 'k+', 'Whisker', 1.5);
set(h, 'LineWidth', 1.6);

% 2) 
x0 = 1;
jitter = 0.08 * (rand(nRuns,1) - 0.5);        
x_scatter = x0 + jitter;

s = scatter(x_scatter, vals, 45, 'filled');
s.MarkerFaceAlpha = 0.75;
s.MarkerEdgeAlpha = 0.75;

% 3) 
plot([0.75 1.25], [mu mu], '-',  'LineWidth', 2.2);   
plot([0.75 1.25], [med med], '--', 'LineWidth', 2.0); 

% 4) 
pad = 0.12 * (mx - mn + eps);
ylow = mn - pad;
yhigh = mx + pad;
ylim([ylow, yhigh]);
xlim([0.6, 1.4]);

% 5) 
xlabel('Independent PSO runs (different random seeds)');
ylabel('Final best fitness (validation MAE, lower is better)');
title(sprintf('Final best fitness across %d random initializations', nRuns));

set(gca, 'XTick', 1, 'XTickLabel', {'PSO'}, 'FontSize', 12, 'LineWidth', 1.2);
grid on;

% 6) 
txt = {
    sprintf('Runs (seeds): %d', nRuns)
    sprintf('Mean ± Std: %.6f ± %.6f', mu, sg)
    sprintf('Median: %.6f', med)
    sprintf('Min / Max: %.6f / %.6f', mn, mx)
    sprintf('IQR: %.6f', iqr_val)
    };


x_text = 1.29;
y_text = yhigh - 0.05*(yhigh-ylow);
text(x_text, y_text, txt, 'HorizontalAlignment','right', ...
    'VerticalAlignment','top', 'FontSize', 11, 'BackgroundColor',[1 1 1], ...
    'EdgeColor',[0.4 0.4 0.4], 'Margin', 8);

% 7) 
legend({'Mean','Median'}, 'Location','southoutside', 'Orientation','horizontal');
fprintf('\n[Supplement] Multi-seed PSO robustness (%d runs)\n', nRuns);
fprintf('Best fitness: mean = %.6f, std = %.6f, min = %.6f, max = %.6f\n', ...
    mean(all_bestfitness), std(all_bestfitness), min(all_bestfitness), max(all_bestfitness));
save('supp_multi_seed_pso.mat', 'seed_list', 'all_curve', 'all_bestfitness');
%% 
%% 

outSobol = supp_Sobol_FigureA(Mdl, x_mu, x_sig, y_mu, y_sig, ...
    train_x_feature_label, 'C:\Users\UserX\Desktop\Train.xlsx', ...
    'p_default', 0.02, 'Nsobol', 1000000, 'TopK', 10, 'Seed', 2000);

