function D = pdist2(X, Y)
% Minimal Euclidean pdist2, supplied because the Octave Forge "statistics"
% package could not be built in this environment.  The official EDOLAB GMPB
% generator calls pdist2 exactly once:
%
%     Shift = (ShiftOffset ./ pdist2(ShiftOffset, zeros(1,Dimension))) .* ShiftSeverity
%
% i.e. with Y a single zero row, so the call reduces to the Euclidean norm of
% each row of X.  This function implements the standard definition
% D(i,j) = || X(i,:) - Y(j,:) ||_2 for the general case.  It is validated in
% codes/gmpb_equivalence.py against the closed form.
  nx = size(X,1); ny = size(Y,1);
  D = zeros(nx, ny);
  for i = 1:nx
    for j = 1:ny
      D(i,j) = sqrt(sum((X(i,:) - Y(j,:)).^2));
    end
  end
end
