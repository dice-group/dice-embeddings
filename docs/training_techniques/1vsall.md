# 1vsAll Training Technique
Steps

    .
    ├── Load & Preprocess input data # G := {(h,r,t)}
    ├── for g in G: 
    ├──     h,r,t :::= g  
    ├──     x:= (h,r), y := one-hot(t) 
    ├──     loss := l(f(x),y)