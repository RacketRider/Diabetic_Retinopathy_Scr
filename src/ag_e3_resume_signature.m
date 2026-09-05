function signature = ag_e3_resume_signature(config)
% AG_E3_RESUME_SIGNATURE Identify every training-relevant E3 configuration field.

identity = config;
for field = ["useGPU" "resumeSignature"]
    if isfield(identity, field), identity = rmfield(identity, field); end
end
signature = jsonencode(identity);
end
