function folders = getDirFileNames(path)

folders = dir(path);

for k = length(folders):-1:1

    % remove folders starting with .
    fname = folders(k).name;
    if fname(1) == '.'
        folders(k) = [ ];
    end
    
    
end
% Just get the names
folders = {folders.name};