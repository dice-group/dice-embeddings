

hr=[1,2,3]
rr=[66]
tr= [3,4,2]


a = tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr))
print(a)

