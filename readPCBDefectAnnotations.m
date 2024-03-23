function out = readPCBDefectAnnotations(fname)

% Copyright 2022 The MathWorks, Inc.

s = readstruct(fname);
out = struct("missing_hole",[],"mouse_bite",[],"open_circuit",[], ...
    "short",[],"spur",[],"spurious_copper",[]);
boxes = arrayfun(@(s) [s.bndbox.xmin s.bndbox.ymin ...
    (s.bndbox.xmax-s.bndbox.xmin) (s.bndbox.ymax-s.bndbox.ymin)], ...
    s.object,UniformOutput=false);
boxes = boxes';
boxes = vertcat(boxes{:});
out.(s.object(1).name) = boxes;
end