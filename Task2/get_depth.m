%change the depth and name according to the test folder
depth = zeros(77,1)
name = 'f777190f-47f5-48af-aade-c73fc0fa8171'
for i = 1:size(depth,1)
    if i<11
        depth(i) = getdepth(strcat('000',int2str(i-1)),name,guidimage,x1,x2,y1,y2)
    elseif i<101
        depth(i) = getdepth(strcat('00',int2str(i-1)),name,guidimage,x1,x2,y1,y2)
    else
        depth(i) = getdepth(strcat('0',int2str(i-1)),name,guidimage,x1,x2,y1,y2)
    end
end

function z = getdepth(filenum,dirname,guidimage,x1,x2,y1,y2)
imagename = strcat(filenum,'_image.jpg')
cloudname = strcat(filenum,'_cloud.bin')
projname = strcat(filenum,'_proj.bin')
files = dir(strcat(dirname,'/',imagename));
idx = randi(numel(files));
snapshot = [files(idx).folder, '/', files(idx).name];
disp(snapshot)

img = imread(snapshot);

xyz = read_bin(strrep(snapshot, imagename, cloudname));
xyz = reshape(xyz, [], 3)';

proj = read_bin(strrep(snapshot, imagename, projname));
proj = reshape(proj, [4, 3])';

uv = proj * [xyz; ones(1, size(xyz, 2))];
uv = uv ./ uv(3, :);
clr = vecnorm(xyz);

ind = find(guidimage == strcat(dirname,'/',filenum))
%x1,x2,y1,y2 are preloaded to MATLAB, and are 2D bounding box coordinates
xr = x2(ind)
xl = x1(ind)
yl = y1(ind)
yu = y2(ind)

%modify the x and y given 2D bbox
x_max = xr
x_min = xl
y_min = yl
y_max = yu
y_mid = y_min+(y_max-y_min)*0.7
sample_num = 20
x_line = linspace(x_min,x_max,sample_num)
point_idx = zeros(sample_num,1)
for i = 1:sample_num
    [~,point_idx(i)] = min(abs(uv(1,:)-x_line(i))+abs(uv(2,:)-y_mid));
end
min_depth = min(clr(point_idx))
threshold = 5
car_depth = []
i1 = 1
i2 = sample_num
for i = 2:sample_num
    if(abs(clr(point_idx(i))-clr(point_idx(i-1)))>threshold)
        i1 = i
        break
    end
end
for i = sample_num-1:1
    if(abs(clr(point_idx(i))-clr(point_idx(i+1)))>threshold)
        i2 = i
        break
    end
end
car_depth = [clr(point_idx(i1)) clr(point_idx(i2))]
z = (car_depth(1)+car_depth(end))/2
end

function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end

function [v, e] = get_bbox(p1, p2)
v = [p1(1), p1(1), p1(1), p1(1), p2(1), p2(1), p2(1), p2(1)
    p1(2), p1(2), p2(2), p2(2), p1(2), p1(2), p2(2), p2(2)
    p1(3), p2(3), p1(3), p2(3), p1(3), p2(3), p1(3), p2(3)];
e = [3, 4, 1, 1, 4, 4, 1, 2, 3, 4, 5, 5, 8, 8
    8, 7, 2, 3, 2, 3, 5, 6, 7, 8, 6, 7, 6, 7];
end


function R = rot(n)
theta = norm(n, 2);
if theta
  n = n / theta;
  K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
  R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
else
  R = eye(3);
end
end
