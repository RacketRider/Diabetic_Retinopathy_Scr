function stem = portableStem(pathValue)
% PORTABLESTEM Extract filename stem robustly across Windows and Linux path separators.
p = char(string(pathValue));
p = strrep(p, '\', '/');
[~, stem, ~] = fileparts(p);
end
