list1=dir('C:\Users\yoons\DeepLearning\Capstone\Detectron2\validation\image\*.jpg');
list2=dir('C:\Users\yoons\DeepLearning\Capstone\Detectron2\validation\annos\*.json');
N=size(list1,1)
%%
% FN={};
% FN(N)={[]};
% for k = 1:N
%     l1=list1(k,1);
% %     fname=[l1.folder,'\',l1.name];
%     fname=['validation\image\',l1.name];
%     FN(k)={fname};
% end
%%
FN={};
FN(N)={[]};
SST={};
SST(N)={[]};
LST={};
LST(N)={[]};
SSO={};
SSO(N)={[]};
LSO={};
LSO(N)={[]};
vest={};
vest(N)={[]};
sling={};
sling(N)={[]};
shorts={};
shorts(N)={[]};
trousers={};
trousers(N)={[]};
skirt={};
skirt(N)={[]};
SSD={};
SSD(N)={[]};
LSD={};
LSD(N)={[]};
vestDress={};
vestDress(N)={[]};
slingDress={};
slingDress(N)={[]};
%%
CT = table(FN',SST',LST',SSO',LSO',vest',sling',shorts',trousers',skirt',SSD',LSD',vestDress',slingDress',...
    'VariableNames',{'imageFilename' 'SST' 'LST' 'SSO' 'LSO' 'vest' 'sling' 'shorts' 'trousers' 'skirt' 'SSD' 'LSD' 'vestDress' 'slingDress'})
%%
for k = 1:N
    l1=list1(k,1);
%     fname=[l1.folder,'\',l1.name];
    fname=['C:\Users\yoons\DeepLearning\Capstone\Detectron2\validation\image\',l1.name];
    CT{k,1}={fname};

    l2=list2(k,1);
    fname=['C:\Users\yoons\DeepLearning\Capstone\Detectron2\validation\annos\',l2.name];
    fid = fopen(fname);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    jdata = jsondecode(str);
    len=length(fieldnames(jdata(1)))-2;
    for l = 1:len
        id=jdata.("item" + num2str(l)).category_id;
        bbox=reshape(jdata.("item" + num2str(l)).bounding_box,1,[]);
        bbox(3:4) = bbox(3:4) - bbox(1:2);
        CT{k,id+1}={bbox};
    end
end
%%
save('DFvalTable',"CT")
%%
% 1: 'short sleeve top'
% 2: 'long sleeve top'
% 3: 'short sleeve outwear'
% 4: 'long sleeve outwear'
% 5: 'vest'
% 6: 'sling'
% 7: 'shorts'
% 8: 'trousers'
% 9: 'skirt'
% 10: 'short sleeve dress'
% 11: 'long sleeve dress'
% 12: 'vest dress'
% 13: 'sling dress'