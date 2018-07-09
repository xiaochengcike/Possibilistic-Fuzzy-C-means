c = 3;              %number of clusters
n = 100;            %number of points
d = 2;              %number of dimensions
% X = rand(n,d);     
a = 2;              %user defined const a in objective function
b = 5;             %user defined const b in objective function
m = 2;              %fuzzifier
eta = 2;            %pcm uncertainty parameter for type 1
epsilon = 10^(-4); %error threshold
max_iter = 1000;   %max number of iterations before exit
eta1 = 2;
eta2 = 3;          %UNCERTAINTY PARAMETERS FOR IT2 PFCM [m1,m2] & [eta1, eta2]
m1 = 2;
m2 = 3;
run_pfcm_type1 = 1;     %Boolean for PFCM Type-1
run_pfcm_intervaltype2 = 1;     %Boolean for PFCM Type-2
labelled_dataset = 0;     %For datasets such as Iris, Breast Cancer etc
unlabelled_dataset = 0;   %For datasets such as Squares3Clust
random_X = 1;             %For randomly generated X
normalrandom_X = 0;       %For normal random X


if random_X
    X = rand(n,d);
end

if normalrandom_X
    X = randn(n,d);
end

if labelled_dataset
    [X,Y]=get_data_iris('iris-dataset.txt',',',5);
    
end

if run_pfcm_intervaltype2
    [V U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
    gamma = calculate_gamma(X,V,U,c,n,m);
    [V U T] = get_final_values_pfcm_intervaltype2(X,c,n,d,m1,m2,eta1,eta2,a,b,gamma,epsilon,max_iter,V);
    graph_2d(X,V,U);
%     graph_2d(X,V,U,'r');
end
if run_pfcm_type1
    [V U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
    gamma = calculate_gamma(X,V,U,c,n,m);
    [V U T] = get_final_values_pfcm_type1(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V);
    figure;
    graph_2d(X,V,U);
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
    V = zeros(c,d);
    U = zeros(c,d);
    err = 10000;
    for iterations = 1:max_iter
        iterations
        U = membership_update(X,V_old,m,c,n);           %U is cxn membership matrix
        T = typicality_update(X,V_old,eta,b,c,n,gamma)    %T is cxn typicality matrix
        V = cluster_update(X,a,b,U,T,m,eta,c,n,d)     %V is cxd matrix containing cluster centers
        err = abs(sum(sum(V-V_old)))
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
        iterations
        U_m1 = membership_update(X,V_old,m1,c,n);           %U is cxn membership matrix
        U_m2 = membership_update(X,V_old,m2,c,n);
        U = (U_m1+U_m2)/2.0;
        [U_L U_R] = get_leftright_fuzzified(U_m1,U_m2,m1,m2);  %U_L and U_R are fuzzified and ordered membership values to be passed to eiasc
        T_1 = typicality_update(X,V_old,eta1,b,c,n,gamma);    %T is cxn typicality matrix
        T_2 = typicality_update(X,V_old,eta2,b,c,n,gamma);     
        [T_L T_R] = get_leftright_fuzzified(T_1,T_2,eta1,eta2); %T_L and T_R are fuzzified and ordered typicality values to be passed to eiasc
        T = (T_1+T_2)/2.0;
        V = eiascc(X, U_L, U_R, T_L, T_R, n, c, d, a, b)     %V is cxd matrix containing cluster centers
        err = abs(sum(sum(V-V_old)))
        if err<epsilon
            break
        end
        V_old = V;
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