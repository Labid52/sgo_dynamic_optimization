% Dump an official GMPB benchmark instance and a set of official fitness probes.
% Calls the unmodified EDOLAB sources in official/.  Usage:
%   octave --no-gui -q gmpb_dump.m CASE SEED OUTDIR
% CASE is one of F1..F12.
args = argv();
caseName = args{1};
seed     = str2double(args{2});
outdir   = args{3};

addpath(fullfile(pwd,'shim'));
addpath(genpath(fullfile(pwd,'official')));
cd(fullfile(pwd,'official'));   % Indicators/Indicators.json is read by a relative path

C = getProConfigurableParameters_GMPB();
% Official CEC competition instance table (competition-hub.github.io/GMPB-Competition/)
switch caseName
  case 'F1',  C.PeakNumber.value=5;   C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F2',  C.PeakNumber.value=10;  C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F3',  C.PeakNumber.value=25;  C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F4',  C.PeakNumber.value=50;  C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F5',  C.PeakNumber.value=100; C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F6',  C.PeakNumber.value=10;  C.ChangeFrequency.value=2500; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F7',  C.PeakNumber.value=10;  C.ChangeFrequency.value=1000; C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F8',  C.PeakNumber.value=10;  C.ChangeFrequency.value=500;  C.Dimension.value=5;  C.ShiftSeverity.value=1;
  case 'F9',  C.PeakNumber.value=10;  C.ChangeFrequency.value=5000; C.Dimension.value=10; C.ShiftSeverity.value=1;
  case 'F10', C.PeakNumber.value=10;  C.ChangeFrequency.value=5000; C.Dimension.value=20; C.ShiftSeverity.value=1;
  case 'F11', C.PeakNumber.value=10;  C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=2;
  case 'F12', C.PeakNumber.value=10;  C.ChangeFrequency.value=5000; C.Dimension.value=5;  C.ShiftSeverity.value=5;
  otherwise, error('unknown case %s', caseName);
end

rand('twister', seed); randn('state', seed);   % Octave analogue of rng(seed)
P = BenchmarkGenerator_GMPB('GMPB', C);

% ---- flatten the environment state --------------------------------------
d = P.Dimension; m = P.PeakNumber; T = P.EnvironmentNumber;
RM = zeros(T, m, d, d);
for t = 1:T
  for k = 1:m
    RM(t,k,:,:) = P.RotationMatrix{t}(:,:,k);
  end
end
PP = zeros(T, m, d); PW = zeros(T, m, d);
for t = 1:T
  PP(t,:,:) = P.PeaksPosition(:,:,t);
  PW(t,:,:) = P.PeaksWidth(:,:,t);
end
ETA = zeros(T, m, 4);
for t = 1:T
  ETA(t,:,:) = P.eta(:,:,t);
end

state = struct();
state.case = caseName; state.seed = seed;
state.Dimension = d; state.PeakNumber = m; state.EnvironmentNumber = T;
state.ChangeFrequency = P.ChangeFrequency; state.ShiftSeverity = P.ShiftSeverity;
state.MinCoordinate = P.MinCoordinate; state.MaxCoordinate = P.MaxCoordinate;
state.MaxEvals = P.MaxEvals;
state.PeaksPosition = PP; state.PeaksWidth = PW;
state.PeaksHeight = P.PeaksHeight; state.PeaksAngle = P.PeaksAngle;
state.tau = P.tau; state.eta = ETA; state.RotationMatrix = RM;
state.OptimumValue = P.OptimumValue; state.OptimumID = P.OptimumID;

if ~exist(outdir,'dir'), mkdir(outdir); end
save('-v7', fullfile(outdir, sprintf('%s_seed%d_state.mat', caseName, seed)), 'state');

% ---- official fitness probes -------------------------------------------
% Deterministic probe points, independent of any optimizer, spread over the box
% and over the environments.  These are the reference values the port must match.
rand('twister', 987654321);
nprobe = 200;
Xp = P.MinCoordinate + (P.MaxCoordinate-P.MinCoordinate)*rand(nprobe, d);
Xp(1,:)  = zeros(1,d);
Xp(2,:)  = P.MinCoordinate*ones(1,d);
Xp(3,:)  = P.MaxCoordinate*ones(1,d);
envs = unique(round(linspace(1, T, 12)));
probe = zeros(numel(envs)*nprobe, 3+d);
r = 0;
for ei = 1:numel(envs)
  P.Environmentcounter = envs(ei);
  for j = 1:nprobe
    v = fitness_GMPB(Xp(j,:), P);
    r = r + 1;
    probe(r,1) = envs(ei); probe(r,2) = j; probe(r,3) = v; probe(r,4:end) = Xp(j,:);
  end
end
csvwrite(fullfile(outdir, sprintf('%s_seed%d_probe.csv', caseName, seed)), probe);
printf('DUMP OK %s seed %d  d=%d m=%d T=%d CF=%d MaxEvals=%d probes=%d\n', ...
       caseName, seed, d, m, T, P.ChangeFrequency, P.MaxEvals, r);
