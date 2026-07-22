clc;
clear;
close all;

%% Initialize Parameters

n = 30;              % Number of particles
t = 100;             % Maximum iterations
d = 2;               % Dimensions

lb = -100;
ub = 100;

w = 0.7;             % Inertia weight
c1 = 1.5;            % Cognitive coefficient
c2 = 1.5;            % Social coefficient

%% Initialize Particle Positions

Positions = rand(n,d)*(ub-lb)+lb;

%% Initialize Velocities

Velocity = zeros(n,d);

disp('Initial Particle Positions:');
disp(Positions);

%% Initial Fitness

Fitness = sum((Positions+0.5).^2, 2);

disp('Initial Fitness:');
disp(Fitness);

%% Personal Best Initialization

Pbest = Positions;
PbestScore = Fitness;

%% Global Best Initialization

[Best_score,index] = min(Fitness);
Best_pos = Positions(index,:);

Convergence = zeros(1,t);

%% Main Loop

for iter = 1:t

    for i = 1:n

        %% Velocity Update
        r1 = rand();
        r2 = rand();

        Velocity(i,:) = w*Velocity(i,:) ...                           % w = inertial weight
                      + c1*r1*(Pbest(i,:)-Positions(i,:)) ...         % c1 = cognitive variable
                      + c2*r2*(Best_pos-Positions(i,:));              % c2 = social variable

        %% Position Update
        Positions(i,:) = Positions(i,:) + Velocity(i,:);

        %% Boundary Check
        Positions(i,:) = max(Positions(i,:),lb);
        Positions(i,:) = min(Positions(i,:),ub);

        %% Fitness Evaluation
        Fitness(i) = sum((Positions(i,:)+0.5).^2);

        %% Update Personal Best
        if Fitness(i) < PbestScore(i)

            Pbest(i,:) = Positions(i,:);
            PbestScore(i) = Fitness(i);

        end

        %% Update Global Best
        if Fitness(i) < Best_score

            Best_score = Fitness(i);
            Best_pos = Positions(i,:);
        end
    end

    Convergence(iter) = Best_score;
    fprintf('Iteration %3d   Best Fitness = %.6f\n',iter,Best_score);

end

%% Final Results
fprintf('Particle Swarm Optimization\n');

fprintf('Best Fitness : %.8f\n',Best_score);

fprintf('Best Position:\n');
disp(Best_pos);
