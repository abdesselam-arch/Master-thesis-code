function [bestX,bestFitness,bestFitnessEvolution,nEval]=VNS_v1(options)
    %--------------------------------------------------------------------------
    % Author: Dr Bouchekara Houssem Rafik El-Hana
    % Date: 15/05/2021
    % VNS version 1
    
    % When using this program please cite:
    % 
    % Ramli, M.A.M., Bouchekara, H.R.E.H., 2020. Solving the problem of
    % large-scale optimal scheduling of distributed energy resources in smart
    % grids using an improved variable neighborhood search. IEEE Access 8,
    % 77321–77335. https://doi.org/10.1109/ACCESS.2020.2986895
    
    % Bouchekara, H.R.E.H., Shahriar, M.S., Javaid, M.S. et al. A variable
    % neighborhood search algorithm for optimal protection coordination of
    % power systems. Soft Comput (2021).
    % https://doi.org/10.1007/s00500-021-05776-4
    
    %--------------------------------------------------------------------------
    DiffFit=rand(1,100);
    TOL=1e-4;
    bestFitnessEvolution=[];
    nEval=0;
    
    %--------------------------------------------------------------------------
    bestX=(options.lb+options.ub)/2;
    bestFitness=feval(options.ObjFunction,bestX);
    nEval=11;
    for k=1:options.MAT_MAXITER
        
        if rand<0.5
            x=bestX+rand(1,options.n).*(bestX-options.lb);
        else
            x=bestX+rand(1,options.n).*(options.ub-bestX);
        end


        [bestX0,bestFitness0,nEval0]=CCM(options,x);
        nEval=nEval+nEval0;
        
        if bestFitness0<bestFitness
            bestX=bestX0;
            bestFitness=bestFitness0;
        end
        %----------------------------------------------------------------------
        bestFitnessEvolution(k)=bestFitness;
        %----------------------------------------------------------------------
        if options.Display_Flag==1
            fprintf('Iteration N° is %g Best Fitness is %g\n',k,bestFitness)
        end
        
        if (nEval>options.MaxFES)&&(options.StopFESFlag==1)
            break
        end
        
        %--------------------------------------------------------------------------------
    end
    end

    function [bestX,bestFitness,nEval]=CCM(options,x)
        nEval=0;
        epsilon=1e-3;
        y=x;
        n=options.n;
        d=rand*eye(n);
        a=-5;
        b=5;
        Fmin=inf;
        for k=1:20
            for j=1:n
                [lambda,fmin,nEval0]=mFibonacci(y(j,:),d(j,:),a,b,options);
                y(j+1,:) = y(j,:) + lambda*d(j,:);
                [y(j+1,:)]=bound(y(j+1,:),options.lb,options.ub);
                nEval=nEval+nEval0;
            end
            x(k+1,:)=y(j+1,:);
            Fmin(k+1)=fmin;

            if abs(max(x(k+1,:)-x(k,:)))<epsilon
                break
            else
                d=diag(x(k+1,:)-x(k,:));
                y(1,:)=x(k+1,:);
            end
        end
        bestX=x(end,:);
        bestFitness=fmin;
    end
    
    function [x]=bound(x,l,u)
        for j = 1:size(x,1)
            x(j,x(j,:)<l)=l(x(j,:)<l);
            x(j,x(j,:)>u)=u(x(j,:)>u);
        end
    end
    
    function [lambda,fmin,nEval]=mFibonacci(y0,d,a,b,options)
        % M. S. Bazaraa, H. D. Sherali, C. M. Shetty, Nonlinear Programming: Theory
        % and Algorithms. Hoboken, NJ, USA: John Wiley & Sons, Inc., 2009, 3rd ed,
        % pp. 354.
        % last modification: 19/03/2020
        nEval=0;
        epsilon=0.001;
        
        % Fibonacci sequence
        F_fb(1) = 1;
        F_fb(2) = 1;
        l = abs(b(1) - a(1))/100;
        for i_fb = 3:100
            F_fb(i_fb) = F_fb(i_fb-1) + F_fb(i_fb-2);
        end
        
        n=numel(nonzeros(F_fb<(b(1) - a(1))/l))+1;
        lambda = a(1) + (F_fb(n-1)/F_fb(n+1))*(b(1) - a(1));
        mu = a(1) + (F_fb(n)/F_fb(n+1))*(b(1) - a(1));

        for k=1:n-2
            f_lambda_k=OptFun((lambda(k)),y0,d,options);
            f_mu_k=OptFun((mu(k)),y0,d,options);
            al=nEval+2;

            if f_lambda_k>f_mu_k
                a(k+1)=lambda(k);
                b(k+1)=b(k);
                lambda(k+1)=mu(k);
                mu(k+1) = a(k+1) + (F_fb(n-k)/F_fb(n-k+1))*(b(k+1) - a(k+1));
                if k==n-2
                    lambda(n)=lambda(n-1);
                    mu(n)=lambda(n-1)+epsilon;
                    f_lambda_n=OptFun((lambda(n)),y0,d,options);
                    f_mu_n=OptFun((mu(n)),y0,d,options);
                    nEval=nEval+2;
                    if f_lambda_n>f_mu_n
                        a(n)=lambda(n);
                        b(n)=b(n-1);
                    elseif f_lambda_n<=f_mu_n
                        a(n)=a(n-1);
                        b(n)=lambda(n);
                    end
                    break
                else
                end
            elseif f_lambda_k<=f_mu_k
                a(k+1)=a(k);
                b(k+1)=mu(k);
                mu(k+1)=lambda(k);
                lambda(k+1) = a(k+1) + (F_fb(n-k-1)/F_fb(n-k+1))*(b(k+1) - a(k+1));
                if k==n-2
                    lambda(n)=lambda(n-1);
                    mu(n)=lambda(n-1)+epsilon;
                    f_lambda_n=OptFun((lambda(n)),y0,d,options);
                    f_mu_n=OptFun((mu(n)),y0,d,options);
                    nEval=nEval+2;
                    if f_lambda_n>f_mu_n
                        a(n)=lambda(n);
                        b(n)=b(n-1);
                    elseif f_lambda_n<=f_mu_n
                        a(n)=a(n-1);
                        b(n)=lambda(n);
                    end
                    break
                else
                end
            end
        end
        lambda=lambda(end);
        fmin=OptFun((lambda(end)),y0,d,options);
    end
    
    function F=OptFun(y,y0,d,options)
        x=y0+d*y;
        F=feval(options.ObjFunction,x);
    end