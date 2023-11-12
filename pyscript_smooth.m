fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
        %% Generated by nipype.interfaces.spm
        if isempty(which('spm')),
             throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));
        end
        [name, version] = spm('ver');
        fprintf('SPM version: %s Release: %s\n',name, version);
        fprintf('SPM path: %s\n', which('spm'));
        spm('Defaults','fMRI');

        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),
           spm_jobman('initcfg');
           spm_get_defaults('cmdline', 1);
        end

        jobs{1}.spm.spatial.smooth.data = {...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,1';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,2';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,3';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,4';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,5';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,6';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,7';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,8';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,9';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,10';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,11';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,12';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,13';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,14';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,15';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,16';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,17';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,18';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,19';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,20';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,21';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,22';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,23';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,24';...
'doi_10.5061_dryad.gt413__v1\nifti_4d.nii,25';...
};
jobs{1}.spm.spatial.smooth.prefix = 's';

        spm_jobman('run', jobs);

        
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;