% c = 3;              %number of clusters
% n = 100;            %number of points
% d = 2;              %number of dimensions
% X = rand(n,d);     
a = 1;              %user defined const a in objective function
b = 5;             %user defined const b in objective function
m = 2;              %fuzzifier only for type 1  
eta = 3;            %pcm uncertainty parameter for type 1
epsilon = 10^(-4); %error threshold
max_iter = 1000;   %max number of iterations before exit
eta1 = 2;
eta2 = 3;          %UNCERTAINTY PARAMETERS FOR IT2 PFCM [m1,m2] & [eta1, eta2]
m1 = 2;
m2 = 3;
run_pfcm_type1 = 0;     %Boolean for PFCM Type-1
run_pfcm_intervaltype2 = 1;     %Boolean for PFCM Type-2
run_pfcm_generaltype2 = 0;   %Boolean for GT2 PFCM
labelled_dataset = 1;     %For datasets such as Iris, Breast Cancer etc
unlabelled_dataset = 0;   %For datasets such as Squares3Clust
random_X = 0;             %For randomly generated X
normalrandom_X = 0;       %For normal random X
graph_2d_bool = 0;
graph_2d_conv = 0;
graph_3d_conv = 0;       %Boolean values for 2D and 3D graphs
graph_3d_bool = 1;
purity_checking_bool=1;%gives the purity checking error rate
f1_score_bool=1;%gives the f1 score
classification_rate_array=[purity_checking_bool f1_score_bool];%this stores which error rates to print
mean_m = 2.8;
std_dev_m = 0.8;          %mean and std deviation for fuzzifier m
mean_eta = 1.7;
std_dev_eta = 0.3;      %mean and std deviation for eta
alpha = 0.1:0.2:0.9;    %alpha values
m_array = 1:1:15;        %range of fuzzifier values
eta_array = 1:1:15;    %range of eta values


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
        [X y]=get_text_data('breastCancer.txt',9);%getting data from breast cancer dataset
%     [X Y]=get_text_data('Bridge.txt',3);%getting data for bridge data dataset
    c = size(unique(y),1);
    d = size(X,2);
    n = size(X,1);
    
end

if run_pfcm_generaltype2
    [V_init U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
    gamma = calculate_gamma(X,V_init,U,c,n,m);
    [V_gt2 U_gt2 T_gt2] = get_final_values_pfcm_generaltype2(X,c,n,d,alpha,m_array, mean_m,std_dev_m,eta_array,mean_eta,std_dev_eta,a,b,gamma,epsilon,max_iter,V_init);
    if graph_2d_bool
        figure;
       graph_2d(X,V_gt2,U_gt2);
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
       graph_2d(X,V_it2,U_it2);
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
    [V_t1 U_t1 T_t1] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V);
   
    if graph_2d_bool
         figure;
       graph_2d(X,V_t1,U_t1);
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

% graph_2d(X,V,U,'r');

function graph_2d(X,V,U)
        color=max((max(U)==U).*linspace(0,1,size(U,1))')';
        color_2=rand(size(V,1),1);
        hold on
        scatter(X(:,1),X(:,2),20, color,"*");
        scatter(V(:,1),V(:,2),100,color_2,'filled');
        set(gca,'Color','k')
        hold off
end

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

function [V U T] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V_init)
    V_old = V_init;
%     V_old = rand(c,d);
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
    end
end

function [V U T] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a,b,gamma,epsilon,max_iter,V_init)
    V_old = V_init;
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
    eta3_right = eta_list(8)              %ASSIGNING ETA VALUES ACCORDING TO ALPHA-CUTS
    eta4_left = eta_list(4)
    eta4_right = eta_list(7)
    eta5_left = eta_list(5)
    eta5_right = eta_list(6)
    
    [V_1 U_1 T_1] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1_left,m1_right,eta1_left,eta1_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_2 U_2 T_2] = get_final_values_pfcm_intervaltype2(X,c,n,d,m2_left,m2_right,eta2_left,eta2_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_3 U_3 T_3] = get_final_values_pfcm_intervaltype2(X,c,n,d,m3_left,m3_right,eta3_left,eta3_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_4 U_4 T_4] = get_final_values_pfcm_intervaltype2(X,c,n,d,m4_left,m4_right,eta4_left,eta4_right,a,b,gamma,epsilon,max_iter,V_init);
    [V_5 U_5 T_5] = get_final_values_pfcm_intervaltype2(X,c,n,d,m5_left,m5_right,eta5_left,eta5_right,a,b,gamma,epsilon,max_iter,V_init);
    
    V = (alpha(1).*V_1 + alpha(2).*V_2 + alpha(3).*V_3 + alpha(4).*V_4 + alpha(5).*V_5)./(sum(alpha));  %Weighted average to compute final centroid values
    U = (alpha(1).*U_1 + alpha(2).*U_2 + alpha(3).*U_3 + alpha(4).*U_4 + alpha(5).*U_5)./(sum(alpha)); %Weighted average to compute final membership values
    T = (alpha(1).*T_1 + alpha(2).*T_2 + alpha(3).*T_3 + alpha(4).*T_4 + alpha(5).*T_5)./(sum(alpha)); %Weighted average to compute final typicality values

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

function [error precision recall f1_score] = classification_rate(X,V,U,y,classification_rate_array)
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