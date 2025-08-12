%% RBC Model – Stationary Distribution via Simulation
% Part 1j : Simulate time series and compute stationary distribution
% Greenwood, Hercowitz & Huffman (1988) with capacity utilisation
% ------------------------------------------------------------------------

clearvars; close all; clc;

%% 1.  Parameters --------------------------------------------------------
beta   = 0.96;      % Discount factor
alpha  = 0.33;      % Capital share
theta  = 1.0;       % Frisch elasticity parameter
gamma  = 2;         % Coefficient of relative risk aversion
B      = 0.075;     % Depreciation-usage scale
omega  = 2;         % Depreciation-usage curvature
A      = 0.592;     % Level of TFP

% Investment-specific shock ε_t ∈ {–Θ, +Θ},  two-state Markov
sigma_data = 0.051;           % std from data   => Θ
lambda     = 0.44;            % first-order serial corr. => π
Theta  = sigma_data;
pi_stay = (1+lambda)/2;       % transition prob.
P  = [ pi_stay  1-pi_stay ;      % 2×2 transition matrix
       1-pi_stay  pi_stay ];
eps_grid = [-Theta ; +Theta];

% Simulation parameters
T_total = 10300;    % Total simulation periods
T_burn = 300;       % Burn-in periods to discard
T_sample = T_total - T_burn;  % Sample periods for analysis

fprintf('=== COMPUTING STATIONARY DISTRIBUTION VIA SIMULATION ===\n\n');

%% 2.  Load Policy Function from Previous Exercise ------------------------
fprintf('Loading policy function from previous exercises...\n');

% Try EGM results first (better for distribution analysis)
policy_loaded = false;
policy_source = '';

try_files = {'rbc_egm_200_results.mat', 'rbc_egm_100_results.mat', 'rbc_egm_results.mat', 'rbc_pfi_results.mat', 'rbc_vfi_results.mat'};
try_sources = {'EGM 200-point', 'EGM 100-point', 'EGM', 'PFI', 'VFI'};

for i = 1:length(try_files)
    if exist(try_files{i}, 'file')
        try
            load(try_files{i}, 'Kpol', 'k_grid');
            if exist('V', 'var')
                load(try_files{i}, 'V'); % Also load value function if available
            end
            policy_source = try_sources{i};
            policy_loaded = true;
            fprintf('Successfully loaded policy function from %s\n', policy_source);
            break;
        catch
            fprintf('Failed to load from %s\n', try_files{i});
        end
    end
end

if ~policy_loaded
    error('Could not find any policy function results. Please run exercise_1b.m, 1d.m, 1e.m, or 1g.m first.');
end

Nk = length(k_grid);
fprintf('Grid size: %d points\n', Nk);
fprintf('Capital grid range: [%.3f, %.3f]\n', k_grid(1), k_grid(end));

%% 3.  Policy Function Interpolation Functions ---------------------------
function k_next = policy_interp(k_current, eps_idx, Kpol, k_grid)
    % Interpolate policy function at arbitrary capital level
    
    if k_current <= k_grid(1)
        k_next = Kpol(1, eps_idx);
    elseif k_current >= k_grid(end)
        k_next = Kpol(end, eps_idx);
    else
        % Linear interpolation
        idx_low = find(k_grid <= k_current, 1, 'last');
        idx_high = idx_low + 1;
        
        if idx_high <= length(k_grid)
            weight = (k_current - k_grid(idx_low)) / (k_grid(idx_high) - k_grid(idx_low));
            k_next = (1 - weight) * Kpol(idx_low, eps_idx) + weight * Kpol(idx_high, eps_idx);
        else
            k_next = Kpol(idx_low, eps_idx);
        end
    end
end

%% 4.  Simulate Markov Chain for Shocks ---------------------------------
fprintf('\n--- Simulating Investment-Specific Shock Process ---\n');

% Initialize shock sequence
eps_sim = zeros(T_total, 1);
eps_idx_sim = zeros(T_total, 1);

% Start with good shock (arbitrary choice)
eps_idx_sim(1) = 2;  % Good shock
eps_sim(1) = eps_grid(2);

% Generate Markov chain
rng(12345);  % Set seed for reproducibility
for t = 2:T_total
    % Draw random number
    u = rand();
    
    % Determine next shock based on transition matrix
    if u <= P(eps_idx_sim(t-1), eps_idx_sim(t-1))
        % Stay in same state
        eps_idx_sim(t) = eps_idx_sim(t-1);
    else
        % Switch to other state
        eps_idx_sim(t) = 3 - eps_idx_sim(t-1);  % Switch between 1 and 2
    end
    
    eps_sim(t) = eps_grid(eps_idx_sim(t));
end

% Analyze shock process
bad_count = sum(eps_idx_sim(T_burn+1:end) == 1);
good_count = sum(eps_idx_sim(T_burn+1:end) == 2);
fprintf('Simulated shock frequencies (excluding burn-in):\n');
fprintf('  Bad shocks:  %d (%.1f%%)\n', bad_count, 100*bad_count/T_sample);
fprintf('  Good shocks: %d (%.1f%%)\n', good_count, 100*good_count/T_sample);
fprintf('  Theoretical: %.1f%% bad, %.1f%% good\n', 50.0, 50.0);

%% 5.  Simulate Capital Path ---------------------------------------------
fprintf('\n--- Simulating Capital Accumulation Path ---\n');

% Initialize capital sequence
k_sim = zeros(T_total, 1);

% Find reasonable starting capital (near middle of policy function range)
ss_errors = abs(Kpol - repmat(k_grid, 1, 2));  % Find steady states
[~, ss_idx_bad] = min(ss_errors(:,1));
[~, ss_idx_good] = min(ss_errors(:,2));
k_start = (k_grid(ss_idx_bad) + k_grid(ss_idx_good)) / 2;
k_sim(1) = k_start;

fprintf('Starting capital: k_0 = %.4f\n', k_start);

% Simulate capital evolution
for t = 2:T_total
    k_current = k_sim(t-1);
    eps_idx = eps_idx_sim(t-1);  % Use lagged shock for policy
    
    % Get next period capital using policy function
    k_sim(t) = policy_interp(k_current, eps_idx, Kpol, k_grid);
    
    % Ensure capital stays within reasonable bounds
    k_sim(t) = max(k_grid(1), min(k_grid(end), k_sim(t)));
    
    % Progress indicator
    if mod(t, 1000) == 0
        fprintf('Simulation progress: %d/%d periods\n', t, T_total);
    end
end

% Analyze capital path
k_sample = k_sim(T_burn+1:end);  % Exclude burn-in
eps_sample = eps_sim(T_burn+1:end);
eps_idx_sample = eps_idx_sim(T_burn+1:end);

fprintf('\nCapital path statistics (excluding burn-in):\n');
mean_k_sample = mean(k_sample);
std_k_sample = std(k_sample);
min_k_sample = min(k_sample);
max_k_sample = max(k_sample);

fprintf('  Mean:     %.4f\n', mean_k_sample);
fprintf('  Std Dev:  %.4f\n', std_k_sample);
fprintf('  Min:      %.4f\n', min_k_sample);
fprintf('  Max:      %.4f\n', max_k_sample);

%% 6.  Compute Simulated Stationary Distribution -------------------------
fprintf('\n--- Computing Simulated Stationary Distribution ---\n');

% Create histogram bins using the same grid as policy function
% Add some padding for values outside the original grid
k_min_sim = min(k_sample);
k_max_sim = max(k_sample);
k_range_sim = k_max_sim - k_min_sim;

% Use policy function grid but extend if necessary
if k_min_sim < k_grid(1) || k_max_sim > k_grid(end)
    % Create extended grid
    k_min_ext = min(k_min_sim - 0.1*k_range_sim, k_grid(1));
    k_max_ext = max(k_max_sim + 0.1*k_range_sim, k_grid(end));
    k_hist_grid = linspace(k_min_ext, k_max_ext, Nk);
    fprintf('Extended grid to accommodate simulated range: [%.3f, %.3f]\n', k_min_ext, k_max_ext);
else
    k_hist_grid = k_grid;
end

% Compute overall distribution
[hist_counts, ~] = histcounts(k_sample, k_hist_grid);
hist_density = hist_counts / (T_sample * (k_hist_grid(2) - k_hist_grid(1)));
k_hist_centers = (k_hist_grid(1:end-1) + k_hist_grid(2:end)) / 2;

% Compute conditional distributions
k_bad = k_sample(eps_idx_sample == 1);
k_good = k_sample(eps_idx_sample == 2);

[hist_bad, ~] = histcounts(k_bad, k_hist_grid);
[hist_good, ~] = histcounts(k_good, k_hist_grid);

hist_density_bad = hist_bad / (length(k_bad) * (k_hist_grid(2) - k_hist_grid(1)));
hist_density_good = hist_good / (length(k_good) * (k_hist_grid(2) - k_hist_grid(1)));

% Find peaks and modes
[max_density, max_idx] = max(hist_density);
k_mode_sim = k_hist_centers(max_idx);

fprintf('Simulated distribution characteristics:\n');
fprintf('  Mode (peak):     k = %.4f with density %.4f\n', k_mode_sim, max_density);
fprintf('  Mean capital:    %.4f\n', mean(k_sample));

% Conditional statistics
if ~isempty(k_bad)
    fprintf('  Conditional mean (bad shock):  %.4f\n', mean(k_bad));
end
if ~isempty(k_good)
    fprintf('  Conditional mean (good shock): %.4f\n', mean(k_good));
end

%% 7.  Load Analytical Results for Comparison ----------------------------
fprintf('\n--- Loading Analytical Results for Comparison ---\n');

try
    load('rbc_stationary_dist.mat', 'marg_k_ss', 'k_grid', 'mean_k_ss');
    
    % Interpolate analytical distribution to simulation grid for comparison
    if exist('marg_k_ss', 'var')
        marg_analytical = interp1(k_grid, marg_k_ss, k_hist_centers, 'linear', 0);
        
        fprintf('Analytical distribution characteristics:\n');
        fprintf('  Mean capital: %.4f\n', mean_k_ss);
        
        [max_anal, max_idx_anal] = max(marg_analytical);
        k_mode_anal = k_hist_centers(max_idx_anal);
        fprintf('  Mode (peak):  k = %.4f with density %.4f\n', k_mode_anal, max_anal);
        
        analytical_loaded = true;
    else
        analytical_loaded = false;
        fprintf('Analytical distribution not found in saved file\n');
    end
catch
    analytical_loaded = false;
    fprintf('Could not load analytical results from rbc_stationary_dist.mat\n');
end

%% 8.  Create Comparison Plots -------------------------------------------
fprintf('\n--- Creating Comparison Plots ---\n');

figure('Position', [100, 100, 1400, 900]);

% Main comparison plot
subplot(2,3,[1,4]);
plot(k_hist_centers, hist_density, 'r-', 'LineWidth', 2.5); hold on;
if analytical_loaded
    plot(k_hist_centers, marg_analytical, 'b--', 'LineWidth', 2);
    legend('Simulated Distribution', 'Analytical Distribution', 'Location', 'best');
else
    legend('Simulated Distribution', 'Location', 'best');
end

xlabel('Capital k', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
title('Stationary Distribution Comparison', 'FontSize', 14);
grid on;
xlim([min(k_hist_centers), max(k_hist_centers)]);

% Time series plot (sample)
subplot(2,3,2);
t_plot = 1:min(1000, T_sample);  % Plot first 1000 periods of sample
plot(t_plot, k_sample(t_plot), 'b-', 'LineWidth', 1);
xlabel('Time (periods)', 'FontSize', 11);
ylabel('Capital k', 'FontSize', 11);
title('Simulated Capital Path (Sample)', 'FontSize', 12);
grid on;

% Shock sequence (sample)
subplot(2,3,3);
plot(t_plot, eps_sample(t_plot), 'g-', 'LineWidth', 1.5);
xlabel('Time (periods)', 'FontSize', 11);
ylabel('Investment Shock ε', 'FontSize', 11);
title('Simulated Shock Process (Sample)', 'FontSize', 12);
ylim([min(eps_grid)-0.01, max(eps_grid)+0.01]);
grid on;

% Conditional distributions
subplot(2,3,5);
if ~isempty(k_bad) && ~isempty(k_good)
    plot(k_hist_centers, hist_density_bad, 'b-', 'LineWidth', 2, 'DisplayName', 'Bad Shock (ε = -Θ)'); hold on;
    plot(k_hist_centers, hist_density_good, 'r--', 'LineWidth', 2, 'DisplayName', 'Good Shock (ε = +Θ)');
    xlabel('Capital k', 'FontSize', 11);
    ylabel('Conditional Density', 'FontSize', 11);
    title('Simulated Conditional Distributions', 'FontSize', 12);
    legend('Location', 'best');
    grid on;
end

% Convergence diagnostic
subplot(2,3,6);
% Compute running mean to show convergence
window_size = 500;
if T_sample >= window_size
    running_mean = zeros(T_sample - window_size + 1, 1);
    for i = 1:length(running_mean)
        running_mean(i) = mean(k_sample(i:i+window_size-1));
    end
    plot(window_size:T_sample, running_mean, 'k-', 'LineWidth', 1.5);
    xlabel('Time (periods)', 'FontSize', 11);
    ylabel('Running Mean Capital', 'FontSize', 11);
    title('Convergence Diagnostic', 'FontSize', 12);
    grid on;
end

sgtitle(sprintf('Stationary Distribution via Simulation (%s) - %d periods', ...
        policy_source, T_sample), 'FontSize', 16, 'FontWeight', 'bold');

%% 9.  Statistical Comparison --------------------------------------------
if analytical_loaded
    fprintf('\n--- Statistical Comparison ---\n');
    
    % Compare means
    mean_diff = abs(mean(k_sample) - mean_k_ss);
    fprintf('Mean capital difference: %.6f\n', mean_diff);
    
    % Compare modes
    mode_diff = abs(k_mode_sim - k_mode_anal);
    fprintf('Mode difference: %.6f\n', mode_diff);
    
    % Compute simple distance measure
    % Normalize both distributions to compare shapes
    sim_normalized = hist_density / sum(hist_density);
    anal_normalized = marg_analytical / sum(marg_analytical);
    
    % Compute total variation distance
    tv_distance = 0.5 * sum(abs(sim_normalized - anal_normalized));
    fprintf('Total variation distance: %.6f\n', tv_distance);
    
    if mean_diff < 0.05 && mode_diff < 0.05
        fprintf('EXCELLENT: Simulation closely matches analytical results!\n');
    elseif mean_diff < 0.1 && mode_diff < 0.1
        fprintf('GOOD: Simulation reasonably matches analytical results\n');
    else
        fprintf('NOTE: Some differences between simulation and analytical results\n');
    end
end

%% 10. Save Results ------------------------------------------------------
save('rbc_simulation_results.mat', 'k_sample', 'eps_sample', 'hist_density', ...
     'k_hist_centers', 'hist_density_bad', 'hist_density_good', 'T_sample', ...
     'mean_k_sample', 'std_k_sample', 'k_mode_sim', 'policy_source');

fprintf('\n=== Exercise 1j Complete ===\n');
fprintf('Simulation results saved to rbc_simulation_results.mat\n');
fprintf('Total simulation time: %d periods (%d burn-in + %d sample)\n', T_total, T_burn, T_sample);