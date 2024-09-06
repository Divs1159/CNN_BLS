clc
clear all
close all

%pn = 'sixray\';
pn = 'C:\Users\fta71\Desktop\BLS\Small_Data\cont';

load featureExtractor.mat

% categories = {'guns', 'wrenches', 'pliers', 'scissors'};

train_x = [];
train_y = [];
test_x = [];
test_y = [];
props = {};
bboxes = {};

% ext_img = [pn '*.jpg'];
ext_img = [pn '*.png']
a = dir(ext_img);
nfile = length(a);

for i=1:nfile
    tic
    fn = a(i).name; 
    img = imread([pn fn]);
    oimg = img;
    H_W = size(img);
    H = H_W(1);
    W = H_W(2);
    i2 = img;

%     samples = [];
    index = 2;
    
    if ismatrix(img) == false
        img = rgb2gray(img);
    end
    
    img = imadjust(img);

    % FOR BETTER PERFORMANCE, USE THIS ST VERSION
   tensors=structureTensor(img,3);
   [s1,s3] = getCoherentOne(tensors);

    % FOR MORE SPEED, USE THIS ST VERSION
%      [s1, ~, s3] = stOld(img,2,1);
     

    if isempty(s3)
        img = mat2gray(s1);
    else
        imagesc(s1+s3);
        img = 1*mat2gray((s1+s3));
    end
    
%     img(:,1:100) = 0;
%     img(:,end-100:end) = 0;
%     img(1:10,:) = 0;
%     img(end-10:end,:) = 0;
    
    imagesc(img);

    img = imbinarize(img, 0.05);
%     figure, imagesc(img);
    img = bwareaopen(img,500);
%     figure, imagesc(img);
    L = bwlabel(img,8);
%     imshow(img)
    az = max(max(L));
    for j = 1:max(max(L))
        m = img;
        m(L~=j) = 0;

        bbox = regionprops(m,'BoundingBox');
        disp(bbox.BoundingBox)
        disp(bbox.BoundingBox(3))

        if (bbox.BoundingBox(3) ~= W && bbox.BoundingBox(4) ~= H)

            m = imcrop(oimg,bbox.BoundingBox); 
        
           %i2 = insertShape(i2,'Rectangle',bbox.BoundingBox,'LineWidth',10);
        
           m = imresize(m, [224 224],'bilinear');
        
        if ismatrix(m)
            m = cat(3,m,m,m);
        end
          props{i,1}= fn;
          props{i,index} = m;
          bboxes{i,1}= fn;
          bboxes{i, index} = bbox.BoundingBox;
          imwrite(m,['C:\Users\fta71\Desktop\BLS\Out\' fn '_' num2str(index) '.jpg'],'JPG');
%         layer = 'avg_pool';
%         featuresTest = activations(classifier2,props{index},layer);
%         featuresTest = squeeze(mean(featuresTest,[1 2]))';
%         samples = [samples; featuresTest];
        index = index + 1;
        end    
    end
%     [coeff,score,latent]=pca(samples);
%     score = (score-min(min(score)))/(max(max(score))-min(min(score)));
%     samples = score;

    
end

save("bboxes.mat", 'bboxes')

%     imshow(m);
%     toc
    
function [cTensor,cTensor2] = getCoherentOne(tensors)
cTensor = [];
cTensor2 = [];
[r,c] = size(tensors);
m = -100000000000000000;
index1 = -1;  
index2 = -1;
for i = 1:r
    for j = 1:c
        if isempty(tensors{i,j}) 
            continue;
        end
        
        t = tensors{i,j};
        [u s v] = svd(t);
        if m < max(max(s))
            cTensor2 = cTensor;
            cTensor = t;
            m = max(max(s));
        end
    end
end

% figure,imagesc(cTensor)
end


function [Sxx, Sxy, Syy] = stOld(I,si,so)
I = double(I);
[m n] = size(I);
 
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 
 
Ix = conv2( conv2(I,gd,'same'),g','same' );
Iy = conv2( conv2(I,gd','same'),g,'same' );
 
Ixx = Ix.^2;
Ixy = Ix.*Iy;
Iyy = Iy.^2;
 
x  = -2*so:2*so;
g  = exp(-0.5*(x/so).^2);
Sxx = conv2( conv2(Ixx,g,'same'),g','same' ); 
Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
Syy = conv2( conv2(Iyy,g,'same'),g','same' );

end
%%
function [tensors] = structureTensor(I,N)

I = double(I);
[m n] = size(I);
si = 1;
so = 1;
tensors = {};
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 

a = zeros(5,5);
a(3,:) = gd;
b = zeros(5,5);
b(:,3) = g;
an = a;
index = 1;

%imshow(a)
%figure
%imshow(b)
gradients = {};

index = 1;
for i = 0:N-1
    angle = (2*180*i)/N;
    a = imrotate(an,angle,'bilinear','crop');
    Ig = conv2( conv2(I,a,'same'),b','same' );
    gradients{index} = Ig;
    index = index + 1;
end

nGradients = length(gradients);

for i = 1:nGradients
    I1 = gradients{i};
    for j = 1:nGradients
        I2 = gradients{j};
        Ixy = I1.*I2;
        Sxy = imdiffusefilt(Ixy);
        x  = -2*so:2*so;
        g  = exp(-0.5*(x/so).^2);
%         Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
        tensors{i,j} = Sxy;
%         imagesc(Sxy)
    end
end
end