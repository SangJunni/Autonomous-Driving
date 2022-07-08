function [ value ] = ana2dig( distance )
if(distance<20)
    value=1023;
elseif(distance>=20&&distance<25)
    value=1023+(distance-20)*(756-1023)/5;
elseif(distance>=25&&distance<30)
    value=756+(distance-25)*(756-400)/5;
elseif(distance>=30&&distance<35)
    value=400+(distance-30)*(400-260)/5;
elseif(distance>=35&&distance<40)
    value=260+(distance-35)*(260-145)/5;
elseif(distance>=40&&distance<45)
    value=145+(distance-40)*(145-92)/5;
elseif(distance>=45&&distance<50)
    value=92+(distance-45)*(92-74)/5;
elseif(distance>=50&&distance<55)
    value=74+(distance-50)*(74-60)/5;
elseif(distance>=55&&distance<60)
    value=60+(distance-55)*(60-48)/5;
else
    value=0;
end
end
