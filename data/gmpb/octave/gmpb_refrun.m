% Deterministic reference run through the OFFICIAL fitness.m wrapper.
% A fixed, optimizer-free sequence of query points is fed to the official
% budgeted evaluator, and the resulting CurrentError / Ebbc / offline error are
% dumped.  The Python port must reproduce these exactly.
%   octave --no-gui -q gmpb_refrun.m CASE SEED NEVAL BATCH OUTDIR
args = argv();
caseName = args{1}; seed = str2double(args{2});
nEval = str2double(args{3}); batch = str2double(args{4}); outdir = args{5};

addpath(fullfile(pwd,'shim'));
addpath(genpath(fullfile(pwd,'official')));
cd(fullfile(pwd,'official'));

C = getProConfigurableParameters_GMPB();
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
end
% shorten the run so the reference is cheap but still crosses many changes
C.EnvironmentNumber.value = ceil(nEval / C.ChangeFrequency.value);

rand('twister', seed); randn('state', seed);
P = BenchmarkGenerator_GMPB('GMPB', C);

% ---- dump the exact landscape this reference run used ------------------
d = P.Dimension; m = P.PeakNumber; T = P.EnvironmentNumber;
RM = zeros(T, m, d, d);
for t = 1:T
  for k = 1:m
    RM(t,k,:,:) = P.RotationMatrix{t}(:,:,k);
  end
end
PP = zeros(T, m, d); PW = zeros(T, m, d); ETA = zeros(T, m, 4);
for t = 1:T
  PP(t,:,:) = P.PeaksPosition(:,:,t);
  PW(t,:,:) = P.PeaksWidth(:,:,t);
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
save('-v7', fullfile(outdir, sprintf('%s_seed%d_b%d_refstate.mat', caseName, seed, batch)), 'state');

% Fixed query stream, reproducible in Python from the same integer recipe.
d = P.Dimension;
Q = zeros(nEval, d);
for i = 1:nEval
  for j = 1:d
    Q(i,j) = P.MinCoordinate + (P.MaxCoordinate-P.MinCoordinate) * ...
             mod(sin(i*12.9898 + j*78.233) * 43758.5453, 1);
  end
end

vals = NaN(nEval,1);
i = 1;
while i <= nEval && P.FE < P.MaxEvals
  hi = min(i+batch-1, nEval);
  [r, P] = fitness(Q(i:hi,:), P);
  vals(i:hi) = r;
  nOK = sum(~isnan(r));
  if P.RecentChange == 1
    P.RecentChange = 0;      % the algorithm acknowledges the change
  end
  i = i + max(nOK,1);
  if nOK == 0
    i = i - 1 + 1;           % nothing consumed; loop continues after ack
  end
end

out = struct();
out.case = caseName; out.seed = seed; out.batch = batch; out.nEval = nEval;
out.Dimension = d; out.ChangeFrequency = P.ChangeFrequency;
out.EnvironmentNumber = P.EnvironmentNumber; out.MaxEvals = P.MaxEvals;
out.FE = P.FE; out.Environmentcounter = P.Environmentcounter;
out.CurrentError = P.CurrentError; out.Ebbc = P.Ebbc;
out.OfflineError = mean(P.CurrentError); out.EbbcMean = mean(P.Ebbc);
out.Q = Q; out.vals = vals;
if ~exist(outdir,'dir'), mkdir(outdir); end
save('-v7', fullfile(outdir, sprintf('%s_seed%d_b%d_refrun.mat', caseName, seed, batch)), 'out');
printf('REFRUN OK %s seed %d batch %d  FE=%d env=%d Eo=%.15g Ebbc=%.15g\n', ...
       caseName, seed, batch, P.FE, P.Environmentcounter, mean(P.CurrentError), mean(P.Ebbc));
