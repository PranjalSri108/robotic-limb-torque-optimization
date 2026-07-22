clc;
clear;
close all;

% Initialize the Gray Wolf populations [Initialize positions for each search agent]
n = 6;        % No. of seach agents
t = 100;      % Max no. of itiretions
d = 2;        % Dimensions
lb = -100;    % LowerBound
ub = 100;     % UpperBound

Positions = rand(n,d)*(ub-lb)+lb;

disp('Initial Positions of Grey Wolves:');
disp(Positions);

% Calculate Fitness of each Search agent 
Fitness = sum((Positions + 0.5).^2, 2);

disp ('Initial Fitness fo Grey Wolves:');
disp(Fitness);

% Compare Fitness values and select best three Gray Wolves
% Alpha, Beta, Delta initialization
Alpha_pos = zeros(1,d);
Alpha_score = inf;

Beta_pos = zeros(1,d);
Beta_score = inf;

Delta_pos = zeros(1,d);
Delta_score = inf;

Convergence = zeros(1,t);

for iter = 1:t

    % Linearly decreasing parameter
    a = 2 - 2*iter/t;

    %% Reset leaders
    Alpha_score = inf;
    Beta_score  = inf;
    Delta_score = inf;

    %% Evaluate wolves
    for i = 1:n

        Fitness(i) = sum((Positions(i,:) + 0.5).^2);

        if Fitness(i) < Alpha_score

            Delta_score = Beta_score;
            Delta_pos   = Beta_pos;

            Beta_score  = Alpha_score;
            Beta_pos    = Alpha_pos;

            Alpha_score = Fitness(i);
            Alpha_pos   = Positions(i,:);

        elseif Fitness(i) < Beta_score

            Delta_score = Beta_score;
            Delta_pos   = Beta_pos;

            Beta_score = Fitness(i);
            Beta_pos   = Positions(i,:);

        elseif Fitness(i) < Delta_score

            Delta_score = Fitness(i);
            Delta_pos   = Positions(i,:);

        end
    end

    %% Update positions
    for i = 1:n

        for j = 1:d

            %% Alpha
            r1 = rand();
            r2 = rand();

            A1 = 2*a*r1 - a;
            C1 = 2*r2;

            D_alpha = abs(C1*Alpha_pos(j)-Positions(i,j));
            X1 = Alpha_pos(j)-A1*D_alpha;

            %% Beta
            r1 = rand();
            r2 = rand();

            A2 = 2*a*r1-a;
            C2 = 2*r2;

            D_beta = abs(C2*Beta_pos(j)-Positions(i,j));
            X2 = Beta_pos(j)-A2*D_beta;

            %% Delta
            r1 = rand();
            r2 = rand();

            A3 = 2*a*r1-a;
            C3 = 2*r2;

            D_delta = abs(C3*Delta_pos(j)-Positions(i,j));
            X3 = Delta_pos(j)-A3*D_delta;

            %% New position
            Positions(i,j) = (X1+X2+X3)/3;

        end

        %% Boundary handling
        Positions(i,:) = max(Positions(i,:),lb);
        Positions(i,:) = min(Positions(i,:),ub);

    end

    Convergence(iter) = Alpha_score;

    fprintf('Iteration %3d   Best Fitness = %.6f\n',iter,Alpha_score);

end

fprintf('Alpha Fitness : %.6f\n', Alpha_score);
fprintf('Alpha Position:\n');
disp(Alpha_pos);

fprintf('Beta Fitness  : %.6f\n', Beta_score);
fprintf('Beta Position :\n');
disp(Beta_pos);

fprintf('Delta Fitness : %.6f\n', Delta_score);
fprintf('Delta Position:\n');
disp(Delta_pos);
