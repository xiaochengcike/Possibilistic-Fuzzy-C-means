c = 3;              %number of clusters
n = 100;            %number of points
d = 2;              %number of dimensions
X = randn(n,d);     
a = 2;              %user defined const a in objective function
b = 5;             %user defined const b in objective function
m = 2;              %fuzzifier
eta = 2;            %pcm uncertainty parameter
epsilon = 10^(-4); %error threshold
max_iter = 1000;   %max number of iterations before exit



[V_init U] = get_final_values_fcm(X,c,n,d,m,epsilon,max_iter);
gamma = calculate_gamma(X,V,U,c,n,m);
[V U T] = get_final_values_pfcm(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V_init);
graph_2d(X,V,U);

function graph_2d(X,V,U)
        color=max((max(U)==U).*linspace(0,1,size(U,1))')';
        color_2=rand(size(V,1),1);
        hold on
        scatter(X(:,1),X(:,2),20, color,"*");
        scatter(V(:,1),V(:,2),100,color_2,'filled');
        set(gca,'Color','k')
        hold off
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

function [V U T] = get_final_values_pfcm(X,c,n,d,m,eta,a,b,gamma,epsilon,max_iter,V_init)
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

function y_all = eiascc(X, U_L, U_R, n, c, d) %y_all is cxd matrix containing cluster centers
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
            a = u_left*x;
            b = sum(u_left);               %Initialisation for y_left
            L = 0;
            while true
                %%%%%%%STEP 2: COMPUTATION%%%%%%%%%%%%%%%%%
                L = L + 1;
                a = a + x(L)*(u_right(L)-u_left(L));   %Computation for y_left
                b = b + u_right(L)-u_left(L);
                y_l = a/b;
                %%%%%%%%%STEP 3 : TERMINATION %%%%%%%%%%
                if y_l <= x(L+1)
                    break;
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            a = u_right*x;
            b = sum(u_right);               %Initialisation for y_right
            R = n;
            while true
                %%%%%%%%%%COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%
                a = a + x(R)*(u_right(R)-u_left(R));
                b = b + u_right(R)-u_left(R);           %Computation for y_right
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