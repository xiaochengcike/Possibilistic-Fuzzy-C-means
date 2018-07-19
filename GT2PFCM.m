% c = 3;              %number of clusters
% n = 100;            %number of points
% d = 2;              %number of dimensions
% X = rand(n,d);     
a = 10;              %user defined const a in objective function
b = 1;             %user defined const b in objective function
m = 2;              %fuzzifier only for type 1  
eta = 3;            %pcm uncertainty parameter for type 1
epsilon = 10^(-4); %error threshold
max_iter = 1000;   %max number of iterations before exit
eta1 = 3;
eta2 = 5;          %UNCERTAINTY PARAMETERS FOR IT2 PFCM [m1,m2] & [eta1, eta2]
m1 = 2;
m2 = 3;
run_pfcm_type1 = 0;     %Boolean for PFCM Type-1
run_pfcm_intervaltype2 = 0;     %Boolean for PFCM Type-2
run_pfcm_generaltype2 = 1;   %Boolean for GT2 PFCM
labelled_dataset = 0;     %For datasets such as Iris, Breast Cancer etc
image_seg = 0;
unlabelled_dataset = 1;   %For datasets such as Squares3Clust
comparison_m = 0;         %Boolean for checking robustess w.r.t m
comparison_eta = 0;       %Boolean for checking robustness w.r.t eta
comparison_a = 0;         %Boolean for checking robustness w.r.t a
comparison_b = 0;         %Boolean for checking robustness w.r.t b  
random_X = 0;             %For randomly generated X
normalrandom_X = 0;       %For normal random X
graph_2d_bool = 1;
graph_2d_conv = 0;
graph_3d_conv = 0;       %Boolean values for 2D and 3D graphs
graph_3d_bool = 0;
purity_checking_bool=1;%gives the purity checking error rate
f1_score_bool=1;%gives the f1 score
classification_rate_array=[purity_checking_bool f1_score_bool];%this stores which error rates to print
mean_m = 3;
std_dev_m = 0.3;          %mean and std deviation for fuzzifier m
mean_eta = 3.4;
std_dev_eta = 0.7;      %mean and std deviation for eta
alpha = 0.1:0.2:0.9;    %alpha values
m_array = 1:1:15;        %range of fuzzifier values
eta_array = 1:1:15;    %range of eta values

interval = 3;
filename = 's5';
c = 2;      %only for unlabelled datasets and image segmentation
noppr = 1000;
graph_bool_21 = 0; %used for rgb segmentation
K_nearest_neighbor = 5;

if filename(2)=='5'%5 represents that it is a porsche car;
    graph_bool_21=2;
elseif filename(2)=='6'
    graph_bool_21=2;%tells that image is wolf
end

if image_seg
     indices=0;
     [X indices]=get_image_data_v3(filename,interval,c,noppr,graph_bool_21);
    Y=[1:1:c]';
    [V_init U] = get_final_values_fcm(X,c,size(X,1),size(X,2),m,epsilon,max_iter); 
    gamma = calculate_gamma(X,V_init,U,c,size(X,1),m);
%     [V_gt2 U_gt2 T_gt2] = get_final_values_pfcm_generaltype2(X,c,size(X,1),size(X,2),alpha,m_array, mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b,gamma,epsilon,max_iter,V_init);
%     [V_it2 U_it2 T_it2] = get_final_values_pfcm_intervaltype2(X,c,size(X,1),size(X,2),m1,m2,eta1,eta2,a,b,gamma,epsilon,max_iter,V_init);    
    [V_t1 U_t1 T_t1] = get_final_values_pfcm_type1(X,c,size(X,1),size(X,2),m,eta,a,b,gamma,epsilon,max_iter,V_init,graph_2d_conv);    
    img_error= show_image_v3(X,U_t1,filename,interval,1,size(V_t1,1),indices,K_nearest_neighbor,graph_bool_21,noppr);
end

if unlabelled_dataset
    X = importdata('t_shape.txt');
    n = size(X,1);
    d = size(X,2);
end

if random_X
    c = 3;              %number of clusters
    n = 100;            %number of points
    d = 2;              %number of dimensions
    X = rand(n,d);
end

if normalrandom_X
    c = 3;              %number of clusters
    n = 100;            %number of points
    d = 2;              %number of dimensions
    X = randn(n,d);
end

if labelled_dataset
%     [X,y]=get_data_iris('iris-dataset.txt',',',5);
%       [X y]=get_text_data('breastCancer.txt',9);%getting data from breast cancer dataset
%     [X y]=get_text_data('Bridge.txt',3);%getting data for bridge data dataset
%     [X y]=get_wine_data('Wine-Dataset.txt',1);%getting data for bridge data dataset
%     [X y]=get_text_data('diabetes.txt',9);%getting data for bridge data dataset
    [X y]=get_text_data('Seeds.txt',8);
    c = size(unique(y),1);
    d = size(X,2);
    n = size(X,1);
end

if run_pfcm_generaltype2
    [V_init U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter); 
    gamma = calculate_gamma(X,V_init,U,c,n,m);
%     min_error = 100;
%     for mean_loop = mean_m
%         for std_dev_loop = std_dev_m
            [V_gt2 U_gt2 T_gt2] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array, mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b,gamma,epsilon,max_iter,V_init);
%             [error_loop confusion_matrix_loop] = classification_rate(X,V_gt2,U_gt2,y,classification_rate_array);
%             if min_error > error_loop
%                 min_error = error_loop;
%                 V_min_gt2 = V_gt2;
%                 U_min_gt2 = U_gt2;
%                 min_confusion_matrix = confusion_matrix_loop; 
%             end
%         end
%     end
%     min_error
%     min_confusion_matrix
    if graph_2d_bool
        figure;
%        graph_2d(X,V_gt2,U_gt2);
         graph_2d(X,V_gt2,U_gt2,'g',3);
    end
    if graph_3d_bool 
        figure;
        graph_3d(X,V_gt2,U_gt2,'g',200,'o',1);
    end
    if labelled_dataset
        fprintf("The results for General Type-2 are as follows: ");
        classification_rate(X,V_gt2,U_gt2,y,classification_rate_array);
    end
end

if run_pfcm_intervaltype2
    [V U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
    gamma = calculate_gamma(X,V,U,c,n,m);
    [V_it2 U_it2 T_it2] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a,b,gamma,epsilon,max_iter,V);
    
    if graph_2d_bool
        figure;
%        graph_2d(X,V_it2,U_it2);
        graph_2d(X,V_it2,U_it2,'g',2);
    end
    if graph_3d_bool 
        figure;
        graph_3d(X,V_it2,U_it2,'g',200,'o',1);
    end
    if labelled_dataset
        fprintf("The results for Interval Type-2 are as follows: ");
        classification_rate(X,V_it2,U_it2,y,classification_rate_array);
    end
%     graph_2d(X,V,U,'r');
end

if run_pfcm_type1
    [V U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
    gamma = calculate_gamma(X,V,U,c,n,m);
    [V_t1 U_t1 T_t1] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V,graph_2d_conv);
   
    if graph_2d_bool
       figure;
%        graph_2d(X,V_t1,U_t1);
        graph_2d(X,V_t1,U_t1,'g',1);
    end
    if graph_3d_bool
        figure;
        graph_3d(X,V_t1,U_t1,'g',200,'o',1);
    end
    if labelled_dataset
        fprintf("The results for Type-1 are as follows: ");
        classification_rate(X,V_t1,U_t1,y,classification_rate_array);
    end
%     graph_2d(X,V,U,'r');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%VARYING ETA%%%%%%%%%%%%%%%%%%%%%%%%%

if comparison_eta && labelled_dataset  
    error_t1 = [];
    error_it2 = [];
    error_gt2 = [];
    k=1; 
    eta_value = 3:3:30;
    for eta_comp = eta_value
        [V_t1_comp U_t1_comp T_t1_comp] = get_final_values_pfcm_type1(X,c,n,d,m,eta_comp,a,b,gamma,epsilon,max_iter,V,graph_2d_conv);
        [error_t1(k) confusion_t1] = classification_rate(X,V_t1_comp,U_t1_comp,y,classification_rate_array);        
        hold on;
        scatter(eta_comp,error_t1(k),'filled','g');     
        k = k+1;
    end
    
    eta_mean = 3:3:30;
    k = 1;
    for eta_comp = eta_mean
        [V_it2_comp U_it2_comp T_it2_comp] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta_comp-1,eta_comp+1,a,b,gamma,epsilon,max_iter,V);  
        [error_it2(k) confusion_it2] = classification_rate(X,V_it2_comp,U_it2_comp,y,classification_rate_array);
        [V_gt2_comp U_gt2_comp T_gt2_comp] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array,mean_m,std_dev_m,eta_array,eta_comp,std_dev_eta,a,b,gamma,epsilon,max_iter,V); 
        [error_gt2(k) confusion_gt2] = classification_rate(X,V_gt2_comp,U_gt2_comp,y,classification_rate_array);
        hold on;
        scatter(eta_comp,error_it2(k),'filled','r');
        scatter(eta_comp,error_gt2(k),'filled','b');
        k = k+1;
    end
    
    xlabel('Eta');
    ylabel('Error Rate');
    p_t1 = plot(eta_value,error_t1,'g');
    p_it2 = plot(eta_value,error_it2,'r');
    p_gt2 = plot(eta_value,error_gt2,'b');
    lgd = legend([p_t1 p_it2 p_gt2],'Type 1','Interval Type-2','General Type-2');
    lgd.FontSize = 12;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%VARYING ETA%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%VARYING M%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if comparison_m && labelled_dataset  
    error_t1 = [];
    error_it2 = [];
    error_gt2 = [];
    k=1; 
    m_value = 3:3:30;
    figure;
    for m_comp = m_value
        [V_t1_comp U_t1_comp T_t1_comp] = get_final_values_pfcm_type1(X,c,n,d,m_comp,eta,a,b,gamma,epsilon,max_iter,V,graph_2d_conv);
        [error_t1(k) confusion_t1] = classification_rate(X,V_t1_comp,U_t1_comp,y,classification_rate_array);        
        hold on;
        scatter(m_comp,error_t1(k),'filled','g');     
        k = k+1;
    end
    
    m_mean = 3:3:30;
    k = 1;
    for m_comp = m_mean
        [V_it2_comp U_it2_comp T_it2_comp] = get_final_values_pfcm_intervaltype2(X,c,n,d,m_comp-1,m_comp+1,eta1,eta2,a,b,gamma,epsilon,max_iter,V);  
        [error_it2(k) confusion_it2] = classification_rate(X,V_it2_comp,U_it2_comp,y,classification_rate_array);
        [V_gt2_comp U_gt2_comp T_gt2_comp] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array,m_comp,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b,gamma,epsilon,max_iter,V); 
        [error_gt2(k) confusion_gt2] = classification_rate(X,V_gt2_comp,U_gt2_comp,y,classification_rate_array);
        hold on;
        scatter(m_comp,error_it2(k),'filled','r');
        scatter(m_comp,error_gt2(k),'filled','b');
        k = k+1;
    end
    ylim([0 100]);
    xlabel('Fuzzifier m');
    ylabel('Error Rate');
    p_t1 = plot(m_value,error_t1,'g');
    p_it2 = plot(m_value,error_it2,'r');
    p_gt2 = plot(m_value,error_gt2,'b');
    lgd = legend([p_t1 p_it2 p_gt2],'Type 1','Interval Type-2','General Type-2');
    lgd.FontSize = 12;
end

%%%%%%%%%%%%%%%%%%VARYING M%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%VARYING A%%%%%%%%%%%%%%%%%%%%%%%%%
if comparison_a && labelled_dataset  
    error_t1 = [];
    error_it2 = [];
    error_gt2 = [];
    k=1; 
    a_value = 1:3:30;
    figure;
    for a_comp = a_value
        [V_t1_comp U_t1_comp T_t1_comp] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a_comp,b,gamma,epsilon,max_iter,V,graph_2d_conv);
        [error_t1(k) confusion_t1] = classification_rate(X,V_t1_comp,U_t1_comp,y,classification_rate_array);     
        [V_it2_comp U_it2_comp T_it2_comp] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a_comp,b,gamma,epsilon,max_iter,V);  
        [error_it2(k) confusion_it2] = classification_rate(X,V_it2_comp,U_it2_comp,y,classification_rate_array);
        [V_gt2_comp U_gt2_comp T_gt2_comp] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array,mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a_comp,b,gamma,epsilon,max_iter,V); 
        [error_gt2(k) confusion_gt2] = classification_rate(X,V_gt2_comp,U_gt2_comp,y,classification_rate_array);
        hold on;
        scatter(a_comp,error_t1(k),'filled','g');     
        scatter(a_comp,error_it2(k),'filled','r');
        scatter(a_comp,error_gt2(k),'filled','b');
        k = k+1;
    end
    
    ylim([0 20]);
    xlabel('a');
    ylabel('Error Rate');
    p_t1 = plot(a_value,error_t1,'g');
    p_it2 = plot(a_value,error_it2,'r');
    p_gt2 = plot(a_value,error_gt2,'b');
    lgd = legend([p_t1 p_it2 p_gt2],'Type 1','Interval Type-2','General Type-2');
    lgd.FontSize = 12;
end


%%%%%%%%%%%%%%%%%VARYING A %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%VARYING B%%%%%%%%%%%%%%%%%%%%%%%%%%%
if comparison_b && labelled_dataset  
    error_t1 = [];
    error_it2 = [];
    error_gt2 = [];
    k=1; 
    b_value = 1:3:30;
    figure;
    for b_comp = b_value
        [V_t1_comp U_t1_comp T_t1_comp] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b_comp,gamma,epsilon,max_iter,V,graph_2d_conv);
        [error_t1(k) confusion_t1] = classification_rate(X,V_t1_comp,U_t1_comp,y,classification_rate_array);     
        [V_it2_comp U_it2_comp T_it2_comp] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a,b_comp,gamma,epsilon,max_iter,V);  
        [error_it2(k) confusion_it2] = classification_rate(X,V_it2_comp,U_it2_comp,y,classification_rate_array);
%         [V_gt2_comp U_gt2_comp T_gt2_comp] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array,mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b_comp,gamma,epsilon,max_iter,V); 
%         [error_gt2(k) confusion_gt2] = classification_rate(X,V_gt2_comp,U_gt2_comp,y,classification_rate_array);
        hold on;
        scatter(b_comp,error_t1(k),'filled','g');     
        scatter(b_comp,error_it2(k),'filled','r');
%         scatter(b_comp,error_gt2(k),'filled','b');
        k = k+1;
    end
    
%     ylim([0 10]);
    xlabel('b');
    ylabel('Error Rate');
    p_t1 = plot(b_value,error_t1,'g');
    p_it2 = plot(b_value,error_it2,'r');
%     p_gt2 = plot(b_value,error_gt2,'b');
%     lgd = legend([p_t1 p_it2 p_gt2],'Type 1','Interval Type-2','General Type-2');
%     lgd.FontSize = 12;
end


%%%%%%%%%%%%%%%VARYING B%%%%%%%%%%%%%%%%%%%%%%%%%%%
% graph_2d(X,V,U,'r');

% function graph_2d(X,V,U)
%         color=max((max(U)==U).*linspace(0,1,size(U,1))')';
%         color_2=rand(size(V,1),1);
%         hold on
%         scatter(X(:,1),X(:,2),20, color,"*");
%         scatter(V(:,1),V(:,2),100,color_2,'filled');
%         set(gca,'Color','k')
%         hold off
% end

% function p=graph_2d(X,V,U,i_color)%i_color as in input color
%     V=(V-min(X))./(max(X)-min(X))   ; 
%     X=(X-min(X))./(max(X)-min(X));
%     color=max((max(U)==U).*linspace(0,1,size(U,1))')'   ;
%     color=[color color color]-[0 rand() rand()];
%     color_2=rand(size(V,1),1);
%     hold on
%     scatter(X(:,1),X(:,2),200, color,".");
%     p=scatter(V(:,1),V(:,2),200,i_color,"*");
%     xlabel("Feature 1",'FontSize' , 14,'FontWeight' , 'bold');
%     ylabel("Feature 2",'FontSize' , 14,'FontWeight' , 'bold');    
%      set(gca,'Color','k');
%     hold off
% end

function p=graph_2d(X,V,U,i_color,type)%i_color as in input color
    V=(V-min(X))./(max(X)-min(X))   ; 
    X=(X-min(X))./(max(X)-min(X));
    color=max((max(U)==U).*linspace(0,1,size(U,1))')'   ;
    color=[color color color]-[0 rand() rand()];
    color_2=rand(size(V,1),1);
    hold on
    scatter(X(:,1),X(:,2),200, color,".");
    p=scatter(V(:,1),V(:,2),200,i_color,"o",'filled');
    xlabel("Feature 1",'FontSize' , 14,'FontWeight' , 'bold');
    ylabel("Feature 2",'FontSize' , 14,'FontWeight' , 'bold'); 
    set(gca, 'FontName', 'Times New Roman','FontWeight','Bold',  'LineWidth', 1);
    if type==1
        saveas(gca,'clusterswithcenters_type1.png');
    elseif type==2
        saveas(gca,'clusterswithcenters_intervaltype2.png');
    else
        saveas(gca,'clusterswithcenters_generaltype2.png');
    end
%     set(gca,'Color','k');
    hold off
end

function graph_2d_v2(X,V,U,color)
    V=(V-min(X))./(max(X)-min(X))  ;  
    X=(X-min(X))./(max(X)-min(X));
    color_2=zeros(size(V,1),1);
    hold on
    scatter(V(:,1),V(:,2),200,color,'r*');
    hold off
end

function p=graph_2d_y(X,V,U,i_color,y,type)%i_color as in input color
    V=(V-min(X))./(max(X)-min(X))   ; 
    X=(X-min(X))./(max(X)-min(X));
    color=max((max(U)==U).*linspace(0,1,size(U,1))')'   
%     color=[color color color]-[0 rand() rand()];
    colors=[0 0 1; 1 0 1;0 1 1;0 0 0;];
    y=(y-min(y))/(max(y)-min(y));
    unique_y=unique(y);
    all_indices=cell(size(unique_y));
    max_classifications=0;
    for unique_y_indices=1:size(unique_y,1)
        all_indices(unique_y_indices)={find(color==unique_y(unique_y_indices))};
    end
    for permute_y=perms(unique_y)';
        for unique_y_index=1:size(permute_y,1)
            color(cell2mat(all_indices(unique_y_index)))=permute_y(unique_y_index);
        end
        correct_classifications=sum(color==y);
        if max_classifications<correct_classifications
            max_classifications=correct_classifications;
            temp_color=color;
            correct_color_code=permute_y;
        end
    end
    color=[temp_color temp_color temp_color];
    for unique_y_indices=1:size(unique_y,1)
        replace_indices=find(temp_color==unique_y(unique_y_indices));
        color(replace_indices,1)=colors(unique_y_indices,1);
        color(replace_indices,2)=colors(unique_y_indices,2);
        color(replace_indices,3)=colors(unique_y_indices,3);
    end
   
    hold on
    scatter(X(:,1),X(:,2),200, color,".");
    p=scatter(V(:,1),V(:,2),200,i_color,"o",'filled');
    xlabel("Feature 1",'FontSize' , 14,'FontWeight' , 'bold');
    ylabel("Feature 2",'FontSize' , 14,'FontWeight' , 'bold');    
    set(gca, 'FontName', 'Times New Roman','FontSize',14,'FontWeight','Bold',  'LineWidth', 1);
    if type==1
        saveas(gca,'clusterswithcenters_type1.png');
    else
        saveas(gca,'clusterswithcenters_type2.png');
    end
%     set(gca,'Color','k');
    hold off
end

function p=graph_3d(X,V,U,i_color,point_size,point_shape,filled_color)%i_color as in input color
    V=(V-min(X))./(max(X)-min(X))  ;  
    X=(X-min(X))./(max(X)-min(X));
    color=max((max(U)==U).*linspace(0,1,size(U,1))')'   ;
    color=[color color color]-[0 rand() rand()];
    color_2=rand(size(V,1),1);
    
    scatter3(X(:,1),X(:,2),X(:,3),200,color,'.');
     hold on
     if filled_color
        scatter3(V(:,1),V(:,2),V(:,3),point_size,point_shape,i_color,"filled");
     else
        p= scatter3(V(:,1),V(:,2),V(:,3),point_size,point_shape,i_color);
     end
     xlabel("Feature 1",'FontSize' , 16,'FontWeight' , 'bold');
     ylabel("Feature 2",'FontSize' , 16,'FontWeight' , 'bold');
     zlabel("Feature 3",'FontSize' , 16,'FontWeight' , 'bold');
      set(gca, 'FontName', 'Times New Roman','FontWeight','Bold',  'LineWidth', 1);
end

function [V U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter)
    V_old = rand(c,d);
    err = 10000;
    V = zeros(c,d);
    U = zeros(c,n);
    for iterations = 1:max_iter    
        U = membership_update(X,V_old,m,c,n);
        V = cluster_update_fcm(X,U,c,n,d,m);
        err = abs(sum(sum(V-V_old)));
        if err<epsilon
            break
        end
        V_old = V;
%         graph_2d(X,V,U);
    end
end

function V = cluster_update_fcm(X,U,c,n,d,m)
    new_U = U.^m;
    num = zeros(1,d);
    V = zeros(c,d);
    for i=1:c
        for k = 1:n
           num = num + new_U(i,k).*X(k,:);
        end
        V(i,:) = num;
        num = zeros(1,d);
    end
    V = V./sum(new_U,2); 
end

function gamma = calculate_gamma(X,V,U,c,n,m)      %function to calculate cx1 gamma values for pfcm
    new_U = U.^m;
    gamma = zeros(c,1);
    for i=1:c
        gamma(i) = sum((vecnorm((X-V(i,:))').^2).*new_U(i,:));
    end
    gamma = gamma./sum(new_U,2);
end

function [V U T] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V_init,graph_conv)
    V_old = V_init;
%     V_old = rand(c,d);
    V_saved_type1 = V_old;
    V = zeros(c,d);
    U = zeros(c,d);
    err = 10000;
    for iterations = 1:max_iter
%         iterations
        U = membership_update(X,V_old,m,c,n);           %U is cxn membership matrix
        T = typicality_update(X,V_old,eta,b,c,n,gamma);    %T is cxn typicality matrix
        V = cluster_update(X,a,b,U,T,m,eta,c,n,d) ;    %V is cxd matrix containing cluster centers
        err = abs(sum(sum(V-V_old)));
        if err<epsilon
            break
        end
        V_old = V;
        if graph_conv
            hold on
            graph_2d(X,V_old,U,'g',1);
        end
    end
end

function [V U T] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a,b,gamma,epsilon,max_iter,V_init)
    V_old = V_init;
%     V_old = rand(c,d);
%     V_old = [0.8236 0.6660 0.7027 0.5409; 0.1750 0.8944 0.1536 0.6797;
%     0.1636 0.5166 0.9535 0.0366];%for iris dataset
    V_saved_intervaltype2 = V_old;
    V = zeros(c,d);
    U = zeros(c,d);
    err = 10000;
    for iterations = 1:max_iter
%         iterations
        U_m1 = membership_update(X,V_old,m1,c,n);           %U is cxn membership matrix
        U_m2 = membership_update(X,V_old,m2,c,n);
        U = (U_m1+U_m2)/2.0;
        [U_L U_R] = get_leftright_fuzzified(U_m1,U_m2,m1,m2);  %U_L and U_R are fuzzified and ordered membership values to be passed to eiasc
        T_1 = typicality_update(X,V_old,eta1,b,c,n,gamma);    %T is cxn typicality matrix
        T_2 = typicality_update(X,V_old,eta2,b,c,n,gamma);     
        [T_L T_R] = get_leftright_fuzzified(T_1,T_2,eta1,eta2); %T_L and T_R are fuzzified and ordered typicality values to be passed to eiasc
        T = (T_1+T_2)/2.0;
        V = eiascc(X, U_L, U_R, T_L, T_R, n, c, d, a, b) ;    %V is cxd matrix containing cluster centers
        err = abs(sum(sum(V-V_old)));
        if err<epsilon
            break
        end
        V_old = V;
    end
end

function [V U T] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array, mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b,gamma,epsilon,max_iter,V_init) 
    U_fuzz = gaussian(m_array,std_dev_m,mean_m,0);
    U_eta =  gaussian(m_array,std_dev_m,mean_m,0);
    m_list = alpha_cut(m_array,U_fuzz,std_dev_m,mean_m,alpha,0)
    eta_list = alpha_cut(eta_array, U_eta, std_dev_eta, mean_eta, alpha,0)
    m1_left = m_list(1)
    m1_right = m_list(10)
    m2_left = m_list(2)
    m2_right = m_list(9)
    m3_left = m_list(3)
    m3_right = m_list(8)              %ASSIGNING FUZZIFIER VALUES ACCORDING TO ALPHA-CUTS
    m4_left = m_list(4)
    m4_right = m_list(7)
    m5_left = m_list(5)
    m5_right = m_list(6)
    
    eta1_left = eta_list(1)
    eta1_right = eta_list(10)
    eta2_left = eta_list(2)
    eta2_right = eta_list(9)
    eta3_left = eta_list(3)
    eta3_right = eta_list(8)              %ASSIGNING FUZZIFIER VALUES ACCORDING TO ALPHA-CUTS
    eta4_left = eta_list(4)
    eta4_right = eta_list(7)
    eta5_left = eta_list(5)
    eta5_right = eta_list(6)
    
    [V_1 U_1 T_1] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1_left,m1_right,eta1_left,eta1_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_2 U_2 T_2] = get_final_values_pfcm_intervaltype2(X,c,n,d,m2_left,m2_right,eta2_left,eta2_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_3 U_3 T_3] = get_final_values_pfcm_intervaltype2(X,c,n,d,m3_left,m3_right,eta3_left,eta3_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_4 U_4 T_4] = get_final_values_pfcm_intervaltype2(X,c,n,d,m4_left,m4_right,eta4_left,eta4_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_5 U_5 T_5] = get_final_values_pfcm_intervaltype2(X,c,n,d,m5_left,m5_right,eta5_left,eta5_right,a,b,gamma,epsilon,max_iter,V_init);
%     [V_6 U_6 T_6] = get_final_values_pfcm_intervaltype2(X,c,n,d,m6_left,m6_right,eta6_left,eta6_right,a,b,gamma,epsilon,max_iter,V_init);
%     [V_7 U_7 T_7] = get_final_values_pfcm_intervaltype2(X,c,n,d,m7_left,m7_right,eta7_left,eta7_right,a,b,gamma,epsilon,max_iter,V_init);
%     [V_8 U_8 T_8] = get_final_values_pfcm_intervaltype2(X,c,n,d,m8_left,m8_right,eta8_left,eta8_right,a,b,gamma,epsilon,max_iter,V_init);
%     [V_9 U_9 T_9] = get_final_values_pfcm_intervaltype2(X,c,n,d,m9_left,m9_right,eta9_left,eta9_right,a,b,gamma,epsilon,max_iter,V_init);
%     [V_5 U_5 T_5] = get_final_values_pfcm_intervaltype2(X,c,n,d,m5_left,m5_right,eta5_left,eta5_right,a,b,gamma,epsilon,max_iter,V_init);
    
%     V = (alpha(1).*V_1 + alpha(2).*V_2 + alpha(3).*V_3 + alpha(4).*V_4 + alpha(5).*V_5)./(sum(alpha));  %Weighted average to compute final centroid values
%     U = (alpha(1).*U_1 + alpha(2).*U_2 + alpha(3).*U_3 + alpha(4).*U_4 + alpha(5).*U_5)./(sum(alpha)); %Weighted average to compute final membership values
%     T = (alpha(1).*T_1 + alpha(2).*T_2 + alpha(3).*T_3 + alpha(4).*T_4 + alpha(5).*T_5)./(sum(alpha)); %Weighted average to compute final typicality values
        
      V = (alpha(1).*V_1 + alpha(2).*V_2 + alpha(3).*V_3 + alpha(4).*V_4+ alpha(5).*V_5)./(sum(alpha));  %Weighted average to compute final centroid values
      U = (alpha(1).*U_1 + alpha(2).*U_2 + alpha(3).*U_3 + alpha(4).*U_4+ alpha(5).*U_5)./(sum(alpha)); %Weighted average to compute final membership values
      T = (alpha(1).*T_1 + alpha(2).*T_2 + alpha(3).*T_3 + alpha(4).*T_4+ alpha(5).*T_5)./(sum(alpha)); %Weighted average to compute final typicality values
  
end

function Y=gaussian(X,variance,mean,graph_bool)
    Y=gaussmf(X,[variance mean]);%gaussian graph
    if graph_bool
        figure(1);
        plot(X,Y);
    end
end

function [U_left ,U_right]=get_leftright_fuzzified(U1,U2,m1,m2) %W1_all takes the value for U_L and W2_all takes the values for U_R
    u1 = U1.^m1;
    u2 = U2.^m2;
    U_right=((u1>=u2).*(u1))+((u2>u1).*(u2));
    U_left=((~(u1>u2)).*(u1))+((~(u2>=u1)).*(u2));
end

function U = membership_update(X,V,m,c,n)    %U is cxn membership matrix
    U = zeros(c,n);
    for i=1:c       %looping on number of clusters c
        for k=1:n   %looping on number of points n
            num_den = vecnorm(X(k,:)-V(i,:)); %numerator part of denominator
            den_den = vecnorm((X(k,:)-V)');   %denominator part of denominator
            U(i,k) = 1/sum((num_den./den_den).^(2/(m-1)));
        end
    end
end

function T = typicality_update(X,V,eta,b,c,n,gamma)   %T is cxn typicality matrix    %gamma is 1xc row vector
    for i=1:c       %looping on number of clusters c
        for k=1:n   %looping on number of points n
            T(i,k) = 1/(1+(((vecnorm(X(k,:)-V(i,:)).^2)*b/gamma(i))^(1/(eta-1))));
        end
    end
end

function V = cluster_update(X,a,b,U,T,m,eta,c,n,d)   %X is nxd matrix
    new_T = T.^eta;
    new_U = U.^m;
    num = zeros(1,d);
    V = zeros(c,d);
    for i=1:c
        for k = 1:n
           num = num + (a*new_U(i,k)+b*new_T(i,k)).*X(k,:);
        end
        V(i,:) = num;
        num = zeros(1,d);
    end
    V = V./sum(a.*new_U + b.*new_T,2);  
end

function [X Y]=get_data_iris(filename,delimiter,y_column)%gettomg data from a file separated with a delimiter named filename and the labels are in y_columbn
    data=split(importdata(filename),delimiter);
    X=str2double(data(:,1:(y_column-1)));
    y=double(char(data(:,y_column)));
    Y=[];
    u_rows=unique(y,'rows') ;%unique rows
    u_rows_iterator=0;%unique rows iterator
    for rows=u_rows' 
        u_rows_iterator=u_rows_iterator+1;
        equivalence_matrix=sum((y==rows')')';
        Y=[Y;repmat(u_rows_iterator,sum(equivalence_matrix==max(equivalence_matrix )),1)];
    end  
end

function y_all = eiascc(X, U_L, U_R, T_L, T_R, n, c, d, acons, bcons) %y_all is cxd matrix containing cluster centers
    y_all = zeros(c,d);
    for dim = 1:d
        x = X(:,dim);
        %%%%%%STEP 1: SORTING AND INITIALISATION%%%%%%%%%%%%%
        [sortedx sortedindex] = sort(x);
        x = sortedx;                        %x is nx1 column vector consisting feature "dim" of all points
        for clusters = 1:c
            u_left = U_L(clusters,:);
            u_right = U_R(clusters,:);       %Sorting u_left and u_right according to sortedx
            u_left = u_left(sortedindex);
            u_right = u_right(sortedindex);   %u_left and u_right are 1xn row vectors
            T_left = T_L(clusters,:);
            T_right = T_R(clusters,:);       %Sorting t_left and t_right according to sortedx
            T_left = T_left(sortedindex);
            T_right = T_right(sortedindex);   %t_left and t_right are 1xn row vectors
            
            a = (acons*u_left+bcons*T_left)*x;
            b = sum(acons*u_left+bcons*T_left);               %Initialisation for y_left
            L = 0;
            while true
                %%%%%%%STEP 2: COMPUTATION%%%%%%%%%%%%%%%%%
                L = L + 1;
                a = a + x(L)*(acons*(u_right(L)-u_left(L))+bcons*(T_right(L)-T_left(L)));   %Computation for y_left
                b = b + acons*(u_right(L)-u_left(L))+bcons*(T_right(L)-T_left(L));
                y_l = a/b;
                %%%%%%%%%STEP 3 : TERMINATION %%%%%%%%%%
                if y_l <= x(L+1)
                    break;
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            a = (acons*u_right+bcons*T_right)*x;
            b = sum(acons*u_right+bcons*T_right);               %Initialisation for y_right
            R = n;
            while true
                %%%%%%%%%%COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%
                a = a + x(R)*(acons*(u_right(R)-u_left(R))+bcons*(T_right(R)-T_left(R)));
                b = b + acons*(u_right(R)-u_left(R))+bcons*(T_right(R)-T_left(R));           %Computation for y_right
                y_r = a/b;
                R = R - 1;
                %%%%%%%%%%%%%TERMINATION%%%%%%%%%%%%%%
                if y_r >= x(R)
                    break
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            y_all(clusters,dim) = (y_l+y_r)/2;
        end
    end
end

function [tp fp fn tn]= binary_comparison(calculated_labels,y)

        t_labels=calculated_labels==max(calculated_labels);
        t_y=(y==max(y))*2;
        tp=sum((t_labels-t_y)==-1);
        fp=sum((t_labels-t_y)==1);
        fn=sum((t_labels-t_y)==-2);
        tn=sum((t_labels-t_y)==0);
end

function [X Y]=get_text_data(filename,y_column)
    data=importdata(filename);
    Y=data(:,y_column);
    X=data(:,1:(y_column-1));
end

function [X Y]=get_wine_data(filename,y_column)
    data=importdata(filename);
    Y=data(:,1);
    X=data(:,2:end);
end

function [error confusion_matrix] = classification_rate(X,V,U,y,classification_rate_array)
    y_temp=[];
    y_count=0;
    binary_classification=0;
    for unique_y=[unique(y)]'
        y_count=y_count+1;
        y_temp=[y_temp ;repmat(y_count,sum(unique_y==y),1)];
    end
    y=y_temp;
    y_list=[];
    calculated_labels=max((max(U)==U).*linspace(1,size(U,1),size(U,1))')';
    index=0;
    accuracy=0;
    labels_position=zeros(size(unique(y)));
    unique_y=unique(y);
    for unique_y_index=[unique(y)]'
        max_classifications=0;
        for unique_y_inner_index=[unique(y)]'
            correct_classifications=sum(calculated_labels(index+1:index+sum(unique_y(unique_y_index)==y))==unique_y(unique_y_inner_index) );
            if max_classifications<correct_classifications
                max_classifications=correct_classifications;
                labels_position(unique_y_index)=unique_y_inner_index;
            end    
        end 
        index=index+sum(y==unique_y(unique_y_index));%this needs to be updated
    end

    y_temp=zeros(size(y));
    for labels_index=labels_position'
        y_temp(find(calculated_labels==labels_position(labels_index)))=labels_index;
    end   
%     unique(y)';
%     histc(y,unique(y))';
%     unique(calculated_labels)';
%     histc(calculated_labels,unique(calculated_labels))';
%     [a b]=sort(histc(y,unique(y))');
%     [c d]=sort(histc(calculated_labels,unique(calculated_labels))');
%     y_temp=zeros(size(y));
%     for i=1:size(a,2)
%         y_temp=y_temp+(calculated_labels==d(i)).*b(i);
%     end
%     a=(sort(histc(calculated_labels,unique(calculated_labels))')-sort(histc(y,unique(y))'))
%     error=(sum((a>0).*a)/size(y,1))*100        
  
    calculated_labels=y_temp;
   
    if classification_rate_array(1)%first value of classification_rate array telsl about the purity error rate
        error=(1-sum(calculated_labels==y)/size(y,1))*100;
        disp("purity check error rate="+error);
    end

    if classification_rate_array(2)
        unique_y=unique(y);
        confusion_matrix=zeros(size(unique_y,1));
        for unique_y_index=1:size(unique_y,1)
            for unique_y_index_inner=1:size(unique_y,1)
                a=(y==unique_y(unique_y_index));
                b=(calculated_labels==unique_y(unique_y_index_inner));
                tp=binary_comparison(b,a);
%                 confusion_matrix(unique_y(unique_y_index_inner),unique_y(unique_y_index))=binary_comparison( (calculated_labels==unique_y(unique_y_index_inner)), (y==unique_y(unique_y_index)));
                confusion_matrix(unique_y(unique_y_index_inner),unique_y(unique_y_index))=tp;
                
            end        
        end
        confusion_matrix
         if size(unique(calculated_labels),1)==2
            binary_classification=1;
            [tp fp fn tn]=binary_comparison((~(calculated_labels==max(calculated_labels)))*max(calculated_labels)+(calculated_labels==max(calculated_labels))*min(calculated_labels),(~(y==max(y)))*max(y)+(y==max(y))*min(y));
        end
        if binary_classification
            precision=tp/(tp+fp);
            recall=tp/(tp+fn);
            f1_score=2*(precision*recall)/(precision+recall);
            disp("precision="+precision);
            disp("recall="+recall);
            disp("F1 score="+f1_score);
        end
    end
end

function m_list=alpha_cut(m_array,U,standard_deviation,mean,alpha,graph_bool)%returns the values of m for the alpha cuts %0 graph bool means no grpah else graph
    syms m;
    gauss(m)=gaussmf(m,[standard_deviation mean]);
    gaussinv=finverse(gauss);
    F_right=matlabFunction(gaussinv);%F_right is the right side of the function
    F_left=matlabFunction(-(gaussinv-mean)+mean);%F_left is the left side of the function
    X_right=F_right([alpha]);
    X_left=F_left([alpha]);
    if graph_bool
        hold on
        plot(m_array,U);
        plot(X_right,[alpha 1-alpha],"*");
        plot(X_left,[alpha 1-alpha],"*");
        x_line=X_left(1):0.01:X_right(1);
        y_line=repmat(alpha,size(x_line));
        p1=plot(x_line,y_line);
        x_line=X_left(2):0.01:X_right(2);
        y_line=repmat(1-alpha,size(x_line));
        p2=plot(x_line,y_line);
        ylabel('Membership Values of m','FontSize' , 15,'FontWeight' , 'bold');
        xlabel('Fuzzifier (m)','FontSize' , 15,'FontWeight' , 'bold');
        lgd=legend([p1 p2],'alpha cut','1-alpha cut'); 
        lgd.FontSize = 12;
        set([p1 p2],'LineWidth',2);
        set(gca,'fontsize',15);
        hold off
    end
    m_list=[X_left fliplr(X_right)] ;%m_list is 1*(2*(number of alpha cuts)) vectors in ascending order of m
end

function error_rate=show_image_v3(X,U,filename,interval,base_index,c,indices,K_nearest_neighbor,graph_bool_21,noppr)
    color=max((max(U)==U).*linspace(0,1,size(U,1))')'   ;
    color=color*255; 
    orig_all_indices=indices;
    error_rate=-1;%-1 indicates that error rate is not applicable here
    if graph_bool_21==1
        image_seg_scatter_plot(X,color,base_index+2);
        I=imread(strcat(filename,'.jpg'));
        X_whole=uint8(create_image_knn(X,color,indices,I,K_nearest_neighbor));
        figure(base_index);
        imshow(X_whole);
        return;
    end
%     noppr=graph_bool(20)
    I_mask=imread(strcat(filename,'_mask.jpg'));    
    if size(I_mask,3)==3
        I_mask=rgb2gray(I_mask);
    end
    I_mask=double(I_mask);
    if filename(2)=='5'
        I_mask(find(65>I_mask))=0;
        I_mask(find(I_mask>195))=1;
        I_mask(find(((195>=I_mask).*I_mask)>=130))=0.5;
        I_mask(find(((130>=I_mask).*I_mask)>=65))=0.25;
    else
        I_mask(find(100>I_mask))=0;
        I_mask(find(I_mask>200))=1;
        I_mask(find(((200>=I_mask).*I_mask)>=100))=0.5;
    end
    I_mask=uint8(I_mask*255); 
    max_classifications=0;
    unique_color_all=unique(color);
    negative_indices=0;
    while size(unique_color_all)<c
        negative_indices=negative_indices-1;
        unique_color_all=[unique_color_all; negative_indices];
    end
    for unique_color=perms(unique_color_all)'
        correct_classifications=0;
        unique_color_index=0;
        for color_index=1:noppr:size(color)
            unique_color_index=unique_color_index+1
            correct_classifications=correct_classifications+sum(color(color_index:color_index+noppr-1)==unique_color(unique_color_index));
        end
%         correct_classifications=sum(color(1:100)==unique_color(1))+sum(color(101:200)==unique_color(2));%+sum(color(201:300)==unique_color(3));
        if max_classifications<correct_classifications
            max_classifications=correct_classifications;
            correct_colors=unique_color;
        end
    end
    
    
    
    error_rate=(1-max_classifications/size(color,1))*100
    
    confusion_matrix=zeros(size(correct_colors,1),size(correct_colors,1));
    
    for row_index=1:size(confusion_matrix,1)
        color_index=1;
        for column_index=1:size(confusion_matrix,2)
            confusion_matrix(row_index,column_index)=sum(color(color_index:color_index+noppr-1)==correct_colors(row_index));
            color_index=color_index+noppr;
        end
    end
    confusion_matrix

    X_whole=uint8(create_image_knn(X,color,indices,I_mask,K_nearest_neighbor));

      
    indices_list=cell(size(unique_color_all));
    unique_color=unique(X_whole)';
    for unique_color_index=1:size(unique_color,2)
        indices=find(X_whole==unique_color(unique_color_index));
        indices_list(unique_color_index)={indices};
    end
    max_classifications=0;
    confusion_matrix_2=zeros(size(correct_colors,1),size(correct_colors,1));
    if size(unique_color_all,1)==2%This is only for special case for scene 3  not the generalized one because scene 3 mask does not have white color in it
        unique_color_all(2)=128;
    end
    for unique_color=perms(uint8(unique_color_all))'
        correct_classifications=0;
        for unique_color_index=1:size(unique_color,1)
            X_whole(cell2mat(indices_list(unique_color_index)))=unique_color(unique_color_index);
        end
        correct_classifications=sum(sum(X_whole==I_mask));
        if max_classifications<correct_classifications
            max_classifications=correct_classifications;
            X_temp_whole=X_whole;
            color_map=unique_color;
        end
    end
    X_whole=X_temp_whole;
    error_rate_whole=(1-max_classifications/(size(I_mask,1)*size(I_mask,2)))*100
    
       unique_color=unique(X_whole);
      for row_index=1:size(confusion_matrix,1)
          for column_index=1:size(confusion_matrix,2)
              confusion_matrix_2(row_index,column_index)=sum(sum(((X_whole==unique_color(column_index))*2-1)==(I_mask==unique_color(row_index))));             
          end
      end
      confusion_matrix_2
      figure(base_index);
      imshow(X_whole);
    indices_list=cell(size(unique_color_all));
    unique_color=unique(color)';
    for unique_color_index=1:size(unique_color,2)
        indices=find(color==unique_color(unique_color_index));
        indices_list(unique_color_index)={indices};
    end
    max_classifications=0;
      for unique_color=perms(unique_color_all)'
         correct_classifications=0;
         for unique_color_index=1:size(unique_color,1)
                color(cell2mat(indices_list(unique_color_index)))=unique_color(unique_color_index);
         end
         correct_classifications=sum(I_mask(orig_all_indices)==color);
         if  max_classifications<correct_classifications
             max_classifications=correct_classifications;
             color_temp=color;
         end
      end
      max_classifications
      color=color_temp;
      image_seg_scatter_plot(X,color,base_index+2)
end

function image_seg_scatter_plot(X,color,base_index)
    figure(base_index)
    X_norm=(X-min(X))./(max(X)-min(X));
    my_color=['r' 'g' 'b' 'm' 'c' 'w' 'k'];
    points_type=['o' 'x' 's' '+' 'p' 'h','*'];
    unique_color=unique(color);
    for unique_color_index=1:size(unique_color,1)
        input_indices=find(color==unique_color(unique_color_index));
        scatter3(X_norm(input_indices,1),X_norm(input_indices,2),X_norm(input_indices,3),100,my_color(unique_color_index),points_type(unique_color_index));
        hold on
        unique_color_index=unique_color_index+1;
    end
%     scatter3(X_norm(:,1),X_norm(:,2),X_norm(:,3),100,color,'o','filled')
    
    plotcube([1 1 1],[0 0 0],0);
     xlabel("Entropy",'FontSize' , 16,'FontWeight' , 'bold');
     ylabel({"Median                ","Filtered Intensity"},'FontSize' , 16,'FontWeight' , 'bold');
     zlabel("Gaussian Blur",'FontSize' , 16,'FontWeight' , 'bold');
     set(gca, 'FontName', 'Times New Roman','FontSize',16,'FontWeight','Bold',  'LineWidth', 1)
     
end

function [X, all_indices]=get_image_data_v3(filename,interval,c,noppr,graph_bool_21);
    I=imread(strcat(filename,'.jpg'));
    if size(I,3)==3
        I=rgb2gray(I);
    end
    if graph_bool_21 ==1 
       all_indices=ceil(rand(1,ceil(size(I,1)*size(I,2)/10))*(size(I,1)*size(I,2)))';
       I1=imgaussfilt(I,0.5);%standard deviation set to 0.5 gaussian blur
       I2=medfilt2(I,[3 3]);%m*n box is 3*3; median iltering
       I3_temp=entropyfilt(I,true((9)));%entropy filtering
       I3=(I3_temp-min(I3_temp))./(max(I3_temp)-min(I3_temp));
       X=[I1(all_indices) I2(all_indices) I3(all_indices)];
       X=double(X);
        return;
    elseif graph_bool_21==2
       
       X=[];
       I1=imgaussfilt(I,0.5);%standard deviation set to 0.5 gaussian blur
       I2=medfilt2(I,[3 3]);%m*n box is 3*3; median iltering
       I3_temp=entropyfilt(I,true((9)));%entropy filtering
       I3=(I3_temp-min(I3_temp))./(max(I3_temp)-min(I3_temp));
       I1=double(I1);
       I2=double(I2);
       I3=double(I3);
    else
        I=double(I);
        X=[];
        I1=double(imread(strcat(filename,'_gauss.jpg')));%gaussian blur
        I2=double(imread(strcat(filename,'_med.jpg')));%median filtering
        I3=double(imread(strcat(filename,'_entro.jpg')));%entropy
        I1=(I1-min(min(I1)))/(max(max(I1))-min(min(I1)));
        I2=(I2-min(min(I2)))/(max(max(I2))-min(min(I2)));
        I3=(I3-min(min(I3)))/(max(max(I3))-min(min(I3)));
    end
    I_mask=imread(strcat(filename,'_mask.jpg'));    
    if size(I_mask,3)==3
        I_mask=rgb2gray(I_mask);
    end
    I_mask=double(I_mask);
 
    
    if filename(2)=='5'
        I_mask(find(65>I_mask))=0;
        I_mask(find(I_mask>195))=1;
        I_mask(find(((195>=I_mask).*I_mask)>=130))=0.5;
        I_mask(find(((130>=I_mask).*I_mask)>=65))=0.25;
    else
        I_mask(100>I_mask)=0;
        I_mask(find(I_mask>200))=1;
        I_mask(find(((200>=I_mask).*I_mask)>=100))=0.5;
    end
    all_indices=[];
    for unique_pixel=unique(I_mask)'
        X_temp=[];
        indices=find(I_mask==unique_pixel);
        indices_2=ceil(rand(1,noppr)*size(indices,1));clc
        indices_2=indices(indices_2);
        all_indices=[all_indices; indices_2];
        X_temp=[X_temp I1(indices_2)]    ;
        X_temp=[X_temp I2(indices_2)]    ;
        X_temp=[X_temp I3(indices_2)]    ;
        X=[X ;X_temp];
    end 
    if graph_bool_21
        return;
    end
    X=importdata(strcat(filename,'_X.txt'));
    all_indices=importdata(strcat(filename,'_indices.txt'));
    
%     X_norm=(X-min(X))./(max(X)-min(X));
%     X=X_norm;
%     scatter3(X_norm(:,1),X_norm(:,2),X_norm(:,3) )
   
end

function X_whole=create_image_knn(X,color,indices,I_mask,k)
X_whole=zeros(size(I_mask,1),size(I_mask,2));    
column_array=zeros(size(indices,1),1);
row_array=zeros(size(indices,1),1);
for index=1:size(indices,1)
    column_array(index)=floor((indices(index)-0.1)/size(I_mask,1))+1;
    row_array(index)=indices(index)-(column_array(index)-1)*size(I_mask,1);
end
unique_color=unique(color);
    for row_index=1:size(I_mask,1)
        fprintf("%d ",row_index);
        for column_index=1:size(I_mask,2)
%             min_distance=inf;
     
              color_key_list=zeros(size(unique_color));
%             for index=1:size(indices,1)
%                 column=floor((indices(index)-0.1)/size(I_mask,1))+1;
%                 row=indices(index)-(column-1)*size(I_mask,1);
%                 distance=((row_array(index)-row_index)^2+(column_array(index)-column_index)^2)^0.5;
                distance_list=((row_array-row_index).^2 +(column_array-column_index).^2).^(0.5);
%                 distance_list=[distance_list distance];
%                 if min_distance>distance
%                     min_distance=distance;
%                     min_point=index;
%                 end
%             end
%             X_whole(row_index,column_index)=color(min_point);
            [dummy sorted_indices]=mink(distance_list,k);  
            for i=1:k
                color_key_list=color_key_list+(unique_color==color(sorted_indices(i)));
            end
            
            unique_color_index=find(color_key_list==max(color_key_list));
            X_whole(row_index,column_index)=unique_color(unique_color_index(1));
        end
    end    
end

