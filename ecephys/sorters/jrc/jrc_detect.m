function jrc_detect(config_file)
    %paths will be addded to the matlab path using calls in python

    disp(['running `jrc detect ' config_file])
    eval(['jrc detect ' config_file])  % How not to use eval??
    % jrc detect config_file
    disp('done')
