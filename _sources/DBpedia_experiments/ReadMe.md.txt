# Downloading and Preprocessing DBpedia 2022-03 for Knowledge Graph Embeddings

## Download files
Download EN DBpedia 2022 through executing the following commands.
```bash
mkdir FullDBpedia && cd FullDBpedia
query=$(curl -H "Accept:text/sparql" https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2022-03)
files=$(curl -H "Accept: text/csv" --data-urlencode "query=${query}" https://databus.dbpedia.org/repo/sparql | tail -n+2 | sed 's/"//g')
while IFS= read -r file ; do wget $file; done <<< "$files" 
for f in * ; do mv -- "$f" "train_$f" ; done
# DBpedia did not zip these files. Dask throws an error if all files are not in the same format
bzip2 train_ontology--DEV_type\=parsed.nt
bzip2 train_ontology--DEV_type\=parsed_sorted.nt
cd ..
```

```bash
from dask import dataframe as ddf
df = ddf.read_csv('FullDBpedia/train' +'*',delim_whitespace=True,header=None,usecols=[0, 1, 2],names=['subject', 'relation', 'object'],dtype=str)
df=df.compute()
print(len(df)) # 1,127,738,988
df.to_parquet("dbpedia_03_2022.parquet", engine='pyarrow')
```
## Filtering Triples
We are only interested in **particular triples**.
### Remove all literals
df = df[df["object"].str.startswith('<',na=False)]
len(df) =>  799873202
>>> df.relation.value_counts()
<http://dbpedia.org/ontology/wikiPageWikiLink>        243103258
<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>     149152137
<http://www.w3.org/2002/07/owl#sameAs>                139366172
<http://dbpedia.org/property/wikiPageUsesTemplate>     60755075
<http://purl.org/dc/terms/subject>                     34356092
                                                        ...    
<http://dbpedia.org/property/formerTeammates>                 1
<http://dbpedia.org/property/teammate>                        1
<http://dbpedia.org/property/licenceArea>                     1
<http://dbpedia.org/property/protection>                      1
<http://dbpedia.org/property/adress>                          1
Name: relation, Length: 13954, dtype: int64

# Remove triples having a relation starting with **http://dbpedia.org/ontology/wiki**
df = df[~df["relation"].str.startswith('<http://dbpedia.org/ontology/wiki', na=False)]
>>> len(df) 525,065,591

df = df[df["subject"].str.startswith('<http://dbpedia.org/')]
len(df) =>  463,352,549

>>> df.relation.value_counts()
<http://www.w3.org/2002/07/owl#sameAs>                139366172
<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>     121181249
<http://dbpedia.org/property/wikiPageUsesTemplate>     60755075
<http://purl.org/dc/terms/subject>                     34356092
<http://www.w3.org/ns/prov#wasDerivedFrom>             20247486
                                                        ...    
<http://dbpedia.org/property/imitates>                        1
<http://dbpedia.org/property/affiliatedClubs>                 1
<http://dbpedia.org/property/pmid>                            1
<http://dbpedia.org/property/usNcesDistrictId>                1
<http://dbpedia.org/property/minradius>                       1
Name: relation, Length: 13946, dtype: int64

# Remove Wikipage Uese Template
df = df[~df["relation"].str.startswith('<http://dbpedia.org/property/wikiPageUsesTemplate', na=False)]
402,597,474

# Remove xmlns
df = df[~df["relation"].str.startswith('<http://xmlns.com', na=False)]
len(df) 376,581,991

df.to_parquet("dbpedia_only_03_2022", engine='pyarrow') # 376,581,991
#############################################################################



df = df[df["subject"].str.startswith('<http://dbpedia.org/resource/')]
len(df) => 716,559,847

df = df[df["object"].str.startswith('<http://dbpedia.org/resource/',na=False)]
len(df) => 149,586,877


# Remove triples having a relation starting with **http://dbpedia.org/property/wiki**
df = df[~df["relation"].str.startswith('<http://dbpedia.org/property/wiki', na=False)]

len(df) => 735188046

df = df[~df["relation"].str.startswith('<http://dbpedia.org/property/wiki', na=False)]

len(df)  => 88831385


df = df[~df["relation"].str.startswith('<http://purl.org/dc/terms/subject', na=False)]
len(df) => 54475293


df = df[~df["relation"].str.startswith('<http://www.w3.org/2004/02/skos/core#broader', na=False)]
len(df) => 50121317

df = df[~df["relation"].str.startswith('<http://purl.org/linguistics/gold/hypernym', na=False)]
len(df) => 46107647

df = df[~df["relation"].str.startswith('<http://www.w3.org/2004/02/skos/core#related', na=False)]
len(df) => 46107647

df.to_parquet("dbpedia_only_03_2022", engine='pyarrow') # 46062877