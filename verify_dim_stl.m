TR    = stlread('circle_400mm.stl');
verts = TR.Points / 1000;   % now in metres: 0.40 × 0.40 × 0.005

% Verify dimensions
fprintf('X range: %.3f to %.3f m\n', min(verts(:,1)), max(verts(:,1)));
fprintf('Y range: %.3f to %.3f m\n', min(verts(:,2)), max(verts(:,2)));
% Should print approximately -0.200 to +0.200 for both (medium square)