clear all
clc
close all
options.n=2; % dimension of the problem
options.MAT_MAXITER=100; % Maximum number of iterations
options.ObjFunction=@Ackley; % Name of the objective function
options.lb=-32*ones(1,options.n);   % Lower boundaries    [-32, -32]
options.ub=32*ones(1,options.n);    % Upper boundaries    [32, 32]
options.ProblemSize=length(options.ub);    % Number of the decision variables
options.Display_Flag=1; % Flag for displaying results over iterations
options.run_parallel_index=0; % 1 to run the different runs in parallel
options.run=5; % Number of runs
options.MaxFES=1e6; % Maximum number of function evaluations
options.StopFESFlag=0; % Flag for stopping the program if the maximum number of evaluations is exceeded 

if options.run_parallel_index
    stream = RandStream('mrg32k3a');
    parfor index=1:options.run
        set(stream,'Substream',index);
        RandStream.setGlobalStream(stream)
        [bestX, bestF,bestFitnessEvolution]=VNS_v1(options);
        bestX_M(index,:)=bestX;
        Fbest_M(index)=bestF;
        bestFitnessEvolution_M(index,:)=bestFitnessEvolution;
    end
else
    rng('default')
    for index=1:options.run
        [bestX, bestF,bestFitnessEvolution]=VNS_v1(options);
        bestX_M(index,:)=bestX;
        Fbest_M(index)=bestF;
        bestFitnessEvolution_M(index,:)=bestFitnessEvolution;
    end
end


[a,b]=min(Fbest_M);
figure
plot(1:options.MAT_MAXITER,bestFitnessEvolution_M(b,:))
xlabel('Iterations')
ylabel('Fitness')

fprintf(' MIN=%g\n MEAN=%g\n MEDIAN=%g\n MAX=%g\n SD=%g\n',...
         min(Fbest_M),...
         mean(Fbest_M),...
         median(Fbest_M),...
         max(Fbest_M),...
         std(Fbest_M))