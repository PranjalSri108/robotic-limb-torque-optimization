clc;
clear;
close all;

%% Initialize Parameters

n = 30;              % Number of ants
t = 100;             % Maximum iterations
d = 2;               % Dimensions

lb = -100;           % Lower bound
ub = 100;            % Upper bound

ArchiveSize = n;     % Solution archive size
q = 0.5;             % Selection pressure
zeta = 1;            % Sampling parameter

%% Initialize Ant Population

Positions = rand(n,d)*(ub-lb)+lb;

disp('Initial Positions of Ants');
disp(Positions);

%% Calculate Initial Fitness

Fitness = sum((Positions+0.5).^2, 2);

disp('Initial Fitness');
disp(Fitness);

%% Sort Population

[Fitness,index] = sort(Fitness);
Positions = Positions(index,:);

Best_pos = Positions(1,:);
Best_score = Fitness(1);

Convergence = zeros(1,t);

%% Main Loop

for iter = 1:t

    %% Selection Probabilities

    w = zeros(ArchiveSize,1);

    for i = 1:ArchiveSize
        w(i) = (1/(q*ArchiveSize*sqrt(2*pi))) * ...
               exp(-(i-1)^2/(2*(q*ArchiveSize)^2));
    end

    Prob = w/sum(w);

    %% Generate New Ants

    NewPos = zeros(n,d);
    NewFit = zeros(n,1);

    for k = 1:n

        %% Roulette Wheel Selection

        r = rand;
        c = cumsum(Prob);
        index = find(r<=c,1,'first');

        for j = 1:d

            %% Calculate Sigma

            sigma = 0;

            for m = 1:ArchiveSize
                sigma = sigma + abs(Positions(index,j)-Positions(m,j));
            end

            sigma = zeta*sigma/(ArchiveSize-1);

            %% Gaussian Sampling

            NewPos(k,j) = Positions(index,j) + sigma*randn;

        end

        %% Boundary Check

        NewPos(k,:) = max(NewPos(k,:),lb);
        NewPos(k,:) = min(NewPos(k,:),ub);

        %% Fitness

        NewFit(k) = sum((NewPos(k,:)+0.5).^2);

    end

    %% Merge Archive

    Positions = [Positions;NewPos];
    Fitness = [Fitness;NewFit];

    %% Sort Population

    [Fitness,index] = sort(Fitness);
    Positions = Positions(index,:);

    %% Keep Best Archive

    Positions = Positions(1:ArchiveSize,:);
    Fitness = Fitness(1:ArchiveSize);

    %% Best Solution

    Best_pos = Positions(1,:);
    Best_score = Fitness(1);

    Convergence(iter) = Best_score;

    fprintf('Iteration %3d   Best Fitness = %.6f\n',iter,Best_score);
end

%% Final Results
fprintf('Continuous Ant Colony Optimization\n');
fprintf('Best Fitness : %.8f\n',Best_score);
fprintf('Best Position :\n');
disp(Best_pos);
