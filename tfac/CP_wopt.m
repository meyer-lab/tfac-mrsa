%% Housekeeping
clear all;
close all;
clc;

%% Import HDF5 File
arr = h5read("test_tensor.hdf5", "/tensor");

%% Deal With Missing Values
% Zero Out missing values to meet cp_wopt convention
mask = ~isnan(arr);
arr(isnan(arr)) = 0;

%% Implement Factorization
for i = 25:25
    f = cp_wopt(tensor(arr), tensor(mask), i, "skip_zeroing", true, "init", "rand");
    fprintf("R2X Value With %i Components: %f\n", ncomponents(f), calc_R2X(arr, double(full(f))));
    if i >= 4
        writeToDisk(f);
    end
end

%% Helper Functions
function r2x = calc_R2X(orig, reconstructed)
%%% Ensure the passed values are arrays, not tensors
    var1 = var(reconstructed - orig, 0, "all");
    var2 = var(orig, 0, "all");
    
    r2x = 1 - var1/var2;
end

function writeToDisk(ktens)
%%% Write out components to HDF5 files
    ncomps = string(ncomponents(ktens));
    
    %%% Create files
    h5create("Components/cell_comps_" + ncomps + ".hdf5", "/comps", size(ktens.U{1}));
    h5create("Components/gene_comps_" + ncomps + ".hdf5", "/comps", size(ktens.U{2}));
    h5create("Components/measurement_comps_" + ncomps + ".hdf5", "/comps", size(ktens.U{3}));
    
    %%% Write to files
    h5write("Components/cell_comps_" + ncomps + ".hdf5", "/comps", ktens.U{1});
    h5write("Components/gene_comps_" + ncomps + ".hdf5", "/comps", ktens.U{2});
    h5write("Components/measurement_comps_" + ncomps + ".hdf5", "/comps", ktens.U{3});

end




