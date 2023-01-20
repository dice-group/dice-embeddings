# Experiment Setup
Evaluate DistMult, ComplEx, and QMult with PPE, PPE20 and without it.
+ Number of epochs 256
+ Batch size 1024
+ Optimizer Adam with 0.1 learning rate
+ Training strategy KvsAll

## UMLS Results 
### DistMult with 64 Dimension 
```
model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5329754601226994, 'H@3': 0.7615030674846626, 'H@10': 0.9455521472392638, 'MRR': 0.6703433913510307}
Test {'H@1': 0.5393343419062028, 'H@3': 0.7602118003025718, 'H@10': 0.9402420574886535, 'MRR': 0.6735190534452296}

model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5498466257668712, 'H@3': 0.7714723926380368, 'H@10': 0.9486196319018405, 'MRR': 0.68067999258498}
Test {'H@1': 0.5491679273827534, 'H@3': 0.7594553706505295, 'H@10': 0.9387291981845688, 'MRR': 0.6796027875988165

model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9874424846625767, 'H@3': 0.9984662576687117, 'H@10': 1.0, 'MRR': 0.9929999041411042}
Val {'H@1': 0.7369631901840491, 'H@3': 0.879601226993865, 'H@10': 0.9716257668711656, 'MRR': 0.8182854002678525}
Test {'H@1': 0.7549167927382754, 'H@3': 0.8751891074130106, 'H@10': 0.970499243570348, 'MRR': 0.828957584677964}
```
### DistMult with 128 Dimension 
```
model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5230061349693251, 'H@3': 0.7607361963190185, 'H@10': 0.9371165644171779, 'MRR': 0.6624940630286998}
Test {'H@1': 0.527231467473525, 'H@3': 0.7670196671709532, 'H@10': 0.9341906202723147, 'MRR': 0.6653645540012874}

model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5207055214723927, 'H@3': 0.7668711656441718, 'H@10': 0.9401840490797546, 'MRR': 0.6620455452242894}
Test {'H@1': 0.5317700453857791, 'H@3': 0.7655068078668684, 'H@10': 0.93267776096823, 'MRR': 0.6673365758220787}

model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9979869631901841, 'H@3': 0.9999041411042945, 'H@10': 1.0, 'MRR': 0.9989535403885479}
Val {'H@1': 0.7177914110429447, 'H@3': 0.8696319018404908, 'H@10': 0.9693251533742331, 'MRR': 0.8049406490145431}
Test {'H@1': 0.7231467473524962, 'H@3': 0.875945537065053, 'H@10': 0.9636913767019667, 'MRR': 0.8091417957880905}
```
### DistMult with 256 Dimension
```
model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.4831288343558282, 'H@3': 0.7292944785276073, 'H@10': 0.933282208588957, 'MRR': 0.6353092380552692}
Test {'H@1': 0.5030257186081695, 'H@3': 0.7465960665658093, 'H@10': 0.93267776096823, 'MRR': 0.6486946533034595}

model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.4938650306748466, 'H@3': 0.7392638036809815, 'H@10': 0.9325153374233128, 'MRR': 0.6424533429154998}
Test {'H@1': 0.5060514372163388, 'H@3': 0.7443267776096822, 'H@10': 0.93267776096823, 'MRR': 0.6513986032514979}

model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9995207055214724, 'H@3': 0.9999041411042945, 'H@10': 1.0, 'MRR': 0.9997315950920246}
Val {'H@1': 0.7078220858895705, 'H@3': 0.8581288343558282, 'H@10': 0.9601226993865031, 'MRR': 0.7952724284621991}
Test {'H@1': 0.7261724659606656, 'H@3': 0.8585476550680786, 'H@10': 0.9591527987897126, 'MRR': 0.8057830148901028}
```
### ComplEx with 64 Dimension
```
model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5866564417177914, 'H@3': 0.7852760736196319, 'H@10': 0.9409509202453987, 'MRR': 0.7055156077300503}
Test {'H@1': 0.5605143721633888, 'H@3': 0.7829046898638427, 'H@10': 0.9493192133131618, 'MRR': 0.6935203439705802}

model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5874233128834356, 'H@3': 0.7868098159509203, 'H@10': 0.941717791411043, 'MRR': 0.7069724183794057}
Test {'H@1': 0.5620272314674736, 'H@3': 0.7851739788199698, 'H@10': 0.9485627836611196, 'MRR': 0.694568998597717}

model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9930023006134969, 'H@3': 0.9994248466257669, 'H@10': 1.0, 'MRR': 0.996291858384458}
Val {'H@1': 0.7239263803680982, 'H@3': 0.8819018404907976, 'H@10': 0.9700920245398773, 'MRR': 0.8136686103584984}
Test {'H@1': 0.735249621785174, 'H@3': 0.8948562783661119, 'H@10': 0.970499243570348, 'MRR': 0.8232336410344763}
```
### ComplEx with 128 Dimension
```
model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5858895705521472, 'H@3': 0.8013803680981595, 'H@10': 0.9401840490797546, 'MRR': 0.7109291180459015}
Test {'H@1': 0.6142208774583964, 'H@3': 0.8184568835098336, 'H@10': 0.9568835098335855, 'MRR': 0.7336053518077231}

model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5904907975460123, 'H@3': 0.8013803680981595, 'H@10': 0.9401840490797546, 'MRR': 0.7137689111298857}
Test {'H@1': 0.6202723146747352, 'H@3': 0.8177004538577912, 'H@10': 0.9568835098335855, 'MRR': 0.7368114911134722}

model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.7292944785276073, 'H@3': 0.8742331288343558, 'H@10': 0.9578220858895705, 'MRR': 0.8145630708266636}
Test {'H@1': 0.7594553706505295, 'H@3': 0.8993948562783661, 'H@10': 0.9674735249621785, 'MRR': 0.8368564648415941}
```
### ComplEx with 256 Dimension
```
model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5567484662576687, 'H@3': 0.7829754601226994, 'H@10': 0.9309815950920245, 'MRR': 0.6864560608198865}
Test {'H@1': 0.5552193645990923, 'H@3': 0.7844175491679274, 'H@10': 0.9288956127080181, 'MRR': 0.6857856104680489}

model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5567484662576687, 'H@3': 0.7829754601226994, 'H@10': 0.9317484662576687, 'MRR': 0.6868459312442663}
Test {'H@1': 0.5537065052950075, 'H@3': 0.7866868381240545, 'H@10': 0.9288956127080181, 'MRR': 0.6847641502470895}

model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9999041411042945, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9999520705521472}
Val {'H@1': 0.6986196319018405, 'H@3': 0.8650306748466258, 'H@10': 0.9601226993865031, 'MRR': 0.7928410313286781}
Test {'H@1': 0.7125567322239031, 'H@3': 0.8668683812405447, 'H@10': 0.9523449319213313, 'MRR': 0.80033463082176}
```
### QMult with 64 Dimension
```
model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5912576687116564, 'H@3': 0.7921779141104295, 'H@10': 0.9478527607361963, 'MRR': 0.7111221340400986}
Test {'H@1': 0.5847201210287444, 'H@3': 0.81089258698941, 'H@10': 0.9508320726172466, 'MRR': 0.7121469050687815}

model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5897239263803681, 'H@3': 0.7944785276073619, 'H@10': 0.9478527607361963, 'MRR': 0.71098565530222}
Test {'H@1': 0.5847201210287444, 'H@3': 0.813161875945537, 'H@10': 0.9508320726172466, 'MRR': 0.7129394729323592}

model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9905099693251533, 'H@3': 0.9991372699386503, 'H@10': 1.0, 'MRR': 0.9948028502044989}
Val {'H@1': 0.7569018404907976, 'H@3': 0.9072085889570553, 'H@10': 0.9723926380368099, 'MRR': 0.8390807281551501}
Test {'H@1': 0.7534039334341907, 'H@3': 0.8986384266263238, 'H@10': 0.9712556732223904, 'MRR': 0.8353791577233396}
```

### QMult with 128 Dimension
```
model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5789877300613497, 'H@3': 0.7914110429447853, 'H@10': 0.9363496932515337, 'MRR': 0.7030931582189782}
Test {'H@1': 0.6111951588502269, 'H@3': 0.8101361573373677, 'H@10': 0.9500756429652042, 'MRR': 0.7292412305941115}

model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5797546012269938, 'H@3': 0.7921779141104295, 'H@10': 0.9363496932515337, 'MRR': 0.7038890869838106}
Test {'H@1': 0.6089258698940998, 'H@3': 0.8093797276853253, 'H@10': 0.9500756429652042, 'MRR': 0.7281151768841984}

model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9999041411042945, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9999520705521472}
Val {'H@1': 0.700920245398773, 'H@3': 0.8680981595092024, 'H@10': 0.9578220858895705, 'MRR': 0.7946697030939839}
Test {'H@1': 0.7193645990922845, 'H@3': 0.8782148260211801, 'H@10': 0.9644478063540091, 'MRR': 0.8085078161683722}
```

### QMult with 256 Dimension
```
model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5690184049079755, 'H@3': 0.7883435582822086, 'H@10': 0.9378834355828221, 'MRR': 0.6968076682994638}
Test {'H@1': 0.5877458396369137, 'H@3': 0.8101361573373677, 'H@10': 0.9402420574886535, 'MRR': 0.7129540391623068}

model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.5659509202453987, 'H@3': 0.7898773006134969, 'H@10': 0.9378834355828221, 'MRR': 0.6954943118888822}
Test {'H@1': 0.5885022692889561, 'H@3': 0.8116490166414524, 'H@10': 0.9402420574886535, 'MRR': 0.7134733933877069}

model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/UMLS | 
Train {'H@1': 0.9993289877300614, 'H@3': 0.9997124233128835, 'H@10': 1.0, 'MRR': 0.99956384202454}
Val {'H@1': 0.6932515337423313, 'H@3': 0.8466257668711656, 'H@10': 0.9624233128834356, 'MRR': 0.7861739874788883}
Test {'H@1': 0.7337367624810892, 'H@3': 0.8812405446293494, 'H@10': 0.9629349470499243, 'MRR': 0.8167624629286083}
```

## KINSHIP Results 
### DistMult with 64 Dimension
```
model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.8158941947565543, 'H@3': 0.9372659176029963, 'H@10': 0.9897588951310862, 'MRR': 0.8815253966766725}
Val {'H@1': 0.42602996254681647, 'H@3': 0.7059925093632958, 'H@10': 0.9204119850187266, 'MRR': 0.5933659868101923}
Test {'H@1': 0.4371508379888268, 'H@3': 0.7174115456238361, 'H@10': 0.9236499068901304, 'MRR': 0.602243915270553}

model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.86124765917603, 'H@3': 0.9520131086142322, 'H@10': 0.9919241573033708, 'MRR': 0.9111049009062921}
Val {'H@1': 0.47097378277153557, 'H@3': 0.7228464419475655, 'H@10': 0.9213483146067416, 'MRR': 0.622171182773946}
Test {'H@1': 0.48324022346368717, 'H@3': 0.7243947858472998, 'H@10': 0.9324953445065177, 'MRR': 0.6321603722312271}

model:DistMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.7821863295880149, 'H@3': 0.9144428838951311, 'H@10': 0.981683052434457, 'MRR': 0.8557727154122613}
Val {'H@1': 0.5149812734082397, 'H@3': 0.75187265917603, 'H@10': 0.9461610486891385, 'MRR': 0.657602006386226}
Test {'H@1': 0.49162011173184356, 'H@3': 0.7481378026070763, 'H@10': 0.9408752327746741, 'MRR': 0.6455021543123199}
```
### DistMult with 128 Dimension
```
model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.8942532771535581, 'H@3': 0.9730805243445693, 'H@10': 0.9970154494382022, 'MRR': 0.9357521537849334}
Val {'H@1': 0.3089887640449438, 'H@3': 0.5571161048689138, 'H@10': 0.8810861423220974, 'MRR': 0.4800217246566569}
Test {'H@1': 0.30679702048417135, 'H@3': 0.5614525139664804, 'H@10': 0.8836126629422719, 'MRR': 0.4813249082874184}

model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9384363295880149, 'H@3': 0.9863647003745318, 'H@10': 0.9987710674157303, 'MRR': 0.9631762106722604}
Val {'H@1': 0.33848314606741575, 'H@3': 0.5955056179775281, 'H@10': 0.8951310861423221, 'MRR': 0.5094538369318503}
Test {'H@1': 0.33845437616387336, 'H@3': 0.590782122905028, 'H@10': 0.888733705772812, 'MRR': 0.5074385419953258}

model:DistMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.8728347378277154, 'H@3': 0.9569873595505618, 'H@10': 0.994499063670412, 'MRR': 0.918622238769151}
Val {'H@1': 0.42275280898876405, 'H@3': 0.677434456928839, 'H@10': 0.925561797752809, 'MRR': 0.583125472717917}
Test {'H@1': 0.40875232774674114, 'H@3': 0.6568901303538175, 'H@10': 0.9185288640595903, 'MRR': 0.5700127599274185}
```
### DistMult with 256 Dimension
```
model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9218164794007491, 'H@3': 0.9833216292134831, 'H@10': 0.9979517790262172, 'MRR': 0.9535142194163061}
Val {'H@1': 0.2794943820224719, 'H@3': 0.5332397003745318, 'H@10': 0.8539325842696629, 'MRR': 0.4547277190854119}
Test {'H@1': 0.27001862197392923, 'H@3': 0.5176908752327747, 'H@10': 0.861266294227188, 'MRR': 0.44412598596227865}

model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9435861423220974, 'H@3': 0.9885884831460674, 'H@10': 0.9980688202247191, 'MRR': 0.9666370725519489}
Val {'H@1': 0.2949438202247191, 'H@3': 0.5580524344569289, 'H@10': 0.851123595505618, 'MRR': 0.46799628926363246}
Test {'H@1': 0.2914338919925512, 'H@3': 0.5386405959031657, 'H@10': 0.8524208566108007, 'MRR': 0.46410224647744}

model:DistMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9091760299625468, 'H@3': 0.9710908239700374, 'H@10': 0.9959620786516854, 'MRR': 0.9424920344674776}
Val {'H@1': 0.3801498127340824, 'H@3': 0.6198501872659176, 'H@10': 0.9030898876404494, 'MRR': 0.5409493862820872}
Test {'H@1': 0.36685288640595903, 'H@3': 0.6061452513966481, 'H@10': 0.8985102420856611, 'MRR': 0.5311519534111271}
```

### ComplEx with 64 Dimension
```
model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9348665730337079, 'H@3': 0.9894077715355806, 'H@10': 0.9985369850187266, 'MRR': 0.9622547651331017}
Val {'H@1': 0.47284644194756553, 'H@3': 0.7223782771535581, 'H@10': 0.9213483146067416, 'MRR': 0.6261167299872799}
Test {'H@1': 0.48091247672253257, 'H@3': 0.7374301675977654, 'H@10': 0.9352886405959032, 'MRR': 0.6329444565842112}

model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9737827715355806, 'H@3': 0.9948501872659176, 'H@10': 0.9987710674157303, 'MRR': 0.9847176676574454}
Val {'H@1': 0.5397940074906367, 'H@3': 0.773876404494382, 'H@10': 0.9428838951310862, 'MRR': 0.6807248706690409}
Test {'H@1': 0.5470204841713222, 'H@3': 0.7918994413407822, 'H@10': 0.9478584729981379, 'MRR': 0.6888857571588187}

model:ComplEx | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.914501404494382, 'H@3': 0.9799274344569289, 'H@10': 0.9953768726591761, 'MRR': 0.9483207925909584}
Val {'H@1': 0.6451310861423221, 'H@3': 0.8609550561797753, 'H@10': 0.9653558052434457, 'MRR': 0.7646655543507787}
Test {'H@1': 0.6554934823091247, 'H@3': 0.8743016759776536, 'H@10': 0.9725325884543762, 'MRR': 0.7742500243528103}
```
### ComplEx with 128 Dimension
```
model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.31320224719101125, 'H@3': 0.6053370786516854, 'H@10': 0.8904494382022472, 'MRR': 0.4971047827907605}
Test {'H@1': 0.3165735567970205, 'H@3': 0.6108007448789572, 'H@10': 0.8975791433891993, 'MRR': 0.5035881132705424}

model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.3150749063670412, 'H@3': 0.6118913857677902, 'H@10': 0.8923220973782772, 'MRR': 0.5006905088797937}
Test {'H@1': 0.3263500931098696, 'H@3': 0.6210428305400373, 'H@10': 0.8966480446927374, 'MRR': 0.5116079612680183}

model:ComplEx | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9966643258426966, 'H@3': 0.9995318352059925, 'H@10': 1.0, 'MRR': 0.9981634285268416}
Val {'H@1': 0.5823970037453183, 'H@3': 0.8202247191011236, 'H@10': 0.9461610486891385, 'MRR': 0.7155227296412792}
Test {'H@1': 0.5982309124767226, 'H@3': 0.8161080074487895, 'H@10': 0.9506517690875232, 'MRR': 0.7233265897301888}
```
### ComplEx with 256 Dimension
```
model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.3890449438202247, 'H@3': 0.6713483146067416, 'H@10': 0.9176029962546817, 'MRR': 0.561694124411779}
Test {'H@1': 0.3878026070763501, 'H@3': 0.6666666666666666, 'H@10': 0.9124767225325885, 'MRR': 0.5614223795288453}

model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.3895131086142322, 'H@3': 0.6713483146067416, 'H@10': 0.9171348314606742, 'MRR': 0.5623571179983067}
Test {'H@1': 0.39013035381750466, 'H@3': 0.664804469273743, 'H@10': 0.9124767225325885, 'MRR': 0.5626044115334957}

model:ComplEx | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9998244382022472, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9999024656679151}
Val {'H@1': 0.4143258426966292, 'H@3': 0.6938202247191011, 'H@10': 0.923689138576779, 'MRR': 0.5868882072714717}
Test {'H@1': 0.409683426443203, 'H@3': 0.6848230912476723, 'H@10': 0.9250465549348231, 'MRR': 0.5797264645194788}
```

### QMult with 64 Dimension
```
model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9253862359550562, 'H@3': 0.9878862359550562, 'H@10': 0.9986540262172284, 'MRR': 0.9567311887406756}
Val {'H@1': 0.4597378277153558, 'H@3': 0.7448501872659176, 'H@10': 0.9311797752808989, 'MRR': 0.6248134654727795}
Test {'H@1': 0.4581005586592179, 'H@3': 0.7299813780260708, 'H@10': 0.9413407821229051, 'MRR': 0.61993874508256}

model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9676381086142322, 'H@3': 0.9942064606741573, 'H@10': 0.9989466292134831, 'MRR': 0.9811679639761028}
Val {'H@1': 0.5374531835205992, 'H@3': 0.7916666666666666, 'H@10': 0.9480337078651685, 'MRR': 0.6840278225866142}
Test {'H@1': 0.5297951582867784, 'H@3': 0.7849162011173184, 'H@10': 0.9455307262569832, 'MRR': 0.675945453231867}

model:QMult | embedding_dim:64 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9101708801498127, 'H@3': 0.9801029962546817, 'H@10': 0.9962546816479401, 'MRR': 0.9462391779168131}
Val {'H@1': 0.6352996254681648, 'H@3': 0.8628277153558053, 'H@10': 0.9691011235955056, 'MRR': 0.758385608513419}
Test {'H@1': 0.6256983240223464, 'H@3': 0.8589385474860335, 'H@10': 0.9632216014897579, 'MRR': 0.7519670000544425}
```

### QMult with 128 Dimension
```
model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9906367041198502, 'H@3': 0.9995318352059925, 'H@10': 0.9999414794007491, 'MRR': 0.994969134817274}
Val {'H@1': 0.2588951310861423, 'H@3': 0.5482209737827716, 'H@10': 0.8469101123595506, 'MRR': 0.4468355217200476}
Test {'H@1': 0.26582867783985104, 'H@3': 0.5409683426443203, 'H@10': 0.8575418994413407, 'MRR': 0.4475863857274106}

model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.3450374531835206, 'H@3': 0.6245318352059925, 'H@10': 0.8834269662921348, 'MRR': 0.5208688557261348}
Test {'H@1': 0.319366852886406, 'H@3': 0.6201117318435754, 'H@10': 0.8994413407821229, 'MRR': 0.5079274253370254}

model:QMult | embedding_dim:128 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9963717228464419, 'H@3': 0.9996488764044944, 'H@10': 0.9999414794007491, 'MRR': 0.9980073735955056}
Val {'H@1': 0.5814606741573034, 'H@3': 0.8206928838951311, 'H@10': 0.954119850187266, 'MRR': 0.7154886152948697}
Test {'H@1': 0.5679702048417132, 'H@3': 0.7965549348230913, 'H@10': 0.9492551210428305, 'MRR': 0.7006682502642879}
```

### QMult with 256 Dimension
```
model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:[] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.3946629213483146, 'H@3': 0.6704119850187266, 'H@10': 0.8984082397003745, 'MRR': 0.5614266941866363}
Test {'H@1': 0.3850093109869646, 'H@3': 0.6461824953445066, 'H@10': 0.9068901303538175, 'MRR': 0.5534336213129252}

model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE20'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
Val {'H@1': 0.397940074906367, 'H@3': 0.6694756554307116, 'H@10': 0.8979400749063671, 'MRR': 0.5633158887235111}
Test {'H@1': 0.38640595903165736, 'H@3': 0.6485102420856611, 'H@10': 0.9078212290502793, 'MRR': 0.5543166380663103}

model:QMult | embedding_dim:256 | num_epochs:256 | batch_size:1024 | lr:0.1 | callbacks:['PPE'] | scoring_technique:KvsAll | path_dataset_folder:KGs/KINSHIP | 
Train {'H@1': 0.9994733146067416, 'H@3': 0.9998829588014981, 'H@10': 1.0, 'MRR': 0.999686496789727}
Val {'H@1': 0.39232209737827717, 'H@3': 0.670880149812734, 'H@10': 0.8984082397003745, 'MRR': 0.5625041029687458}
Test {'H@1': 0.39664804469273746, 'H@3': 0.6582867783985102, 'H@10': 0.909217877094972, 'MRR': 0.5623474859596793}
```
