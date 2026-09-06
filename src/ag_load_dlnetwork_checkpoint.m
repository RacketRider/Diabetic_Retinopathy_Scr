function net = ag_load_dlnetwork_checkpoint(path)
% AG_LOAD_DLNETWORK_CHECKPOINT Load the first top-level dlnetwork regardless of variable name.

saved = load(path);
fields = fieldnames(saved);
for i = 1:numel(fields)
    candidate = saved.(fields{i});
    if isa(candidate, 'dlnetwork')
        net = candidate;
        return;
    end
end
error('ag:v3:CheckpointVariable', 'No dlnetwork variable in %s.', path);
end
