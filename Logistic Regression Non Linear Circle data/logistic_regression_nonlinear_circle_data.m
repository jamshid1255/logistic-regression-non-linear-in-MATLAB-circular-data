%Logistic regression nonlinear

clear all
close all

load circle_data.mat

figure(1)
hold on
plot(x(1:50,1),x(1:50,2),'ro')
plot(x(51:100,1),x(51:100,2),'bx')

X=[ones(100,1) x];


Y=[zeros(50,1);ones(50,1)];

w=[0.1;0.2;0.3;0.4;0.5];
eta=0.0001;
n_iter=5000;

likelihood=ones(n_iter,1);
n_missclassification=zeros(n_iter,1);

for i=1:n_iter

    %prediction
    
    for j=1:length(X)
        Yhat(j,1)=sigmoid(w(1,1) + w(2,1)*X(j,2) + w(3,1)*X(j,3) +w(4,1)*(X(j,2)^2) + w(5,1)*(X(j,3)^2));
        
        if (Y(j,1)-output_activation_logistic(Yhat(j,1))) ~= 0
            n_missclassification(i,1)=n_missclassification(i,1)+1;
        end
        
        likelihood(i,1)=likelihood(i,1)*Yhat(j,1)^(Y(j,1))*(1-Yhat(j,1))^(1-Y(j,1));
    end
    
    %weight update rule
    
    w(1)=w(1)+eta*sum((Y-Yhat).*X(:,1));
    w(2)=w(2)+eta*sum((Y-Yhat).*X(:,2));
    w(3)=w(3)+eta*sum((Y-Yhat).*X(:,3));
    w(4)=w(4)+eta*sum((Y-Yhat).*(X(:,2).^2));
    w(5)=w(5)+eta*sum((Y-Yhat).*(X(:,3).^2));
                   
end



x1=linspace(-4,4);
x2=linspace(-4,4);

[X1,X2]=meshgrid(x1,x2);

Z=w(1)+w(2)*X1+w(3)*X2 + w(4).*(X1.^2) + w(5).*(X2.^2);

figure(1)

contourf(X1,X2,Z,[0 0])
plot(x(1:50,1),x(1:50,2),'ro')
plot(x(51:100,1),x(51:100,2),'bx')


figure(2)
plot(1:n_iter, n_missclassification)

figure(3)
plot(1:n_iter, -log(likelihood))
