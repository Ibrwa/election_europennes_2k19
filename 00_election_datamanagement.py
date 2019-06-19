# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:04:11 2019

@author: Ibrwa
Prpject: Analyse des resultats de l election presidentielle en France
Last update: 19/06/2019
"""

import pandas as pd 
import matplotlib.pyplot as plt, matplotlib
import seaborn as sns
import folium 
import geopandas as geopd
import numpy as np
import scipy.stats as stats

pd.options.display.max_rows = 999

matplotlib.style.use('ggplot')


## Read the data ###

election = pd.read_excel("resultats-definitifs-par-departement.xls")

### Data Management: On renomme les colonnes qu on va garder ###

election.rename(columns={'Code du département':'code_dep',
                         'Libellé du département':'lib_dep',
                         'Inscrits':'inscrits',
                         'Abstentions':'abstentions',
                         'Votants':'votants',
                         'Blancs':'blancs',
                         'Nuls':'nuls',
                         'Exprimés':'exprimes',
                         'Voix':'LFI',
                         'Unnamed: 48':'LREM',
                         'Unnamed: 97':'PP',
                         'Unnamed: 174':'RN',
                         'Unnamed: 216':'LR',
                         'Unnamed: 223':'EEV'
                         
                         },inplace = True)


#### filtre sur les colonnes  ####

election = election[['code_dep','lib_dep','inscrits','abstentions',
                     'votants','blancs','nuls','exprimes',
                     'LFI','PP','EEV','LREM','LR','RN']]

### Filtre sur la France metropolitaine ###
## On garde uniquement les codes qui commencent par un chiffre ##

filtre_france_metropol = election['code_dep'].str.contains('Z',regex = False)

## Application du filtre ###

election = election[~(filtre_france_metropol)]

### Calcul des colonnes ####

### taux d'abstention ###

election['taux_abstention'] =( election['abstentions'] / election['inscrits'])*100

## stat des pour taux_election

election['taux_abstention'].describe()
fig,ax = plt.subplots(figsize=(10,10))
ax = sns.distplot(election["taux_abstention"])
plt.title("Distribution du taux d'abstention en %")
plt.show()

### check des departements outiliers
election[['code_dep','lib_dep','taux_abstention']].sort_values(by='taux_abstention',
             ascending=False).head(n=3)

#### Ajout de la colonne maximum pour savoir qui a gagné dans chaque departement ###

election['max_votant'] = election[['LFI','PP','EEV','LREM','LR','RN']].apply(np.max,axis=1)

### Ajout d'un tag pour savoir qui a gagné dans chaque departement #####

partis = ['LFI','PP','EEV','LREM','LR','RN']

### Initialisation

election['gagnant'] = np.nan

## Calcul du parti gagant

for parti in partis:
    election.loc[(election['max_votant']==election[parti]),['gagnant']] = parti

###calcul de l'ecart entre les deux partis ####

election['ecart'] = (np.abs(election['RN'] - election['LREM']))*100/election['exprimes']

### On eneleve la colonne max_votant

election.drop(columns=['max_votant'],inplace=True)

### Distribution des ecarts #####

fig,ax = plt.subplots(figsize=(10,10))
ax = sns.distplot(election["ecart"])
plt.title("Ecart entre les votes du RN et de LREM en %")
plt.show()

#### boxplot de la distribution ###

fig,ax = plt.subplots(figsize=(10,10))
ax = sns.boxplot(x="gagnant", y="ecart", data=election)
plt.title("Boxplot des ecart entre les votes du RN et de LREM en %")
ax.text(1,26,'Paris',horizontalalignment='center', color='purple', weight='bold',fontsize=12)
ax.text(1,24.70,'Hauts-de-Seine',horizontalalignment='center', color='purple', weight='bold',fontsize=12)
plt.show()



### check des departements outiliers des ecart pour lrem ###
election[['code_dep','lib_dep','gagnant','ecart']].sort_values(by='ecart',
             ascending=False).head(n=5)

### check de l'galité des moyennes ####

election['gagnant'].value_counts(normalize=True)
election.groupby('gagnant')['ecart'].aggregate(['count','min','mean','std','max'])




stats.ttest_ind(election.loc[election['gagnant']=="RN",['ecart']],
                election.loc[election['gagnant']=="LREM",['ecart']])

stats.ttest_ind(election.loc[election['gagnant']=="RN",['ecart']],
                election.loc[election['gagnant']=="LREM",['ecart']],equal_var=False)

stats.mannwhitneyu(election.loc[election['gagnant']=="RN",['ecart']],
                election.loc[election['gagnant']=="LREM",['ecart']])

stats.kruskal(election.loc[election['gagnant']=="RN",['ecart']],
                election.loc[election['gagnant']=="LREM",['ecart']])

##################### Statistiques sur les partis #######################
data = election;
###calcul des taux sur tous les partis ####

for parti in partis:
    data[parti] = (data[parti]/data['exprimes'])*100
    
## Carte electorale des parties #####

shp = "departements-20170102-shp/departements-20170102.shp"
#Read shapefile using Geopandas
gdf = geopd.read_file(shp)

## on garde uniquement les departements du metropole ###

gdf = gdf[~(gdf["code_insee"].isin (['971','972','973','974','976']))]

## keep only needed columns ###

gdf = gdf[['code_insee','nom','geometry']]

###Update departement Rhone
gdf.loc[gdf['nom']=='Rhône',['code_insee']] = '69'

###Remove metropole Lyon ##
#gdf = gdf.loc[~(gdf['code_insee']=='69M')]
gdf.loc[gdf['code_insee']=='69M',['code_insee']] = '69'

### plot the map de LREM ###
# Create a map centered at the given latitude and longitude
map_fr = folium.Map(location=[45,1], zoom_start=4)
# Add the color for the chloropleth:
map_fr.choropleth(
 geo_data=gdf,
 name='choropleth',
 data=data,
 columns=['code_dep', 'LREM'],
 key_on='feature.properties.code_insee',
nan_fill_opacity=0.4,
 fill_color='YlGn',
 fill_opacity=0.7,
 line_opacity=0.2,
 legend_name='Carte electorale de LREM'
)
map_fr.save('LREM.html')


### plot the map de RN ###
# Create a map centered at the given latitude and longitude
map_fr = folium.Map(location=[45,1], zoom_start=4)
# Add the color for the chloropleth:
map_fr.choropleth(
 geo_data=gdf,
 name='choropleth',
 data=data,
 columns=['code_dep', 'RN'],
 key_on='feature.properties.code_insee',
nan_fill_opacity=0.4,
 fill_color='YlGn',
 fill_opacity=0.7,
 line_opacity=0.2,
 legend_name='Carte electorale du RN'
)
map_fr.save('RN.html')


########### of France with winner by departement colored ########

### Create map of color function  ###

gagnant_df = election[['code_dep','gagnant']]

gdf2 = pd.merge(left = gdf,
                right = gagnant_df,
                left_on = 'code_insee',
                right_on = 'code_dep')

gdf2.drop(columns=['code_dep'],inplace = True)

gdf2['fill_coolor'] = np.nan
gdf2.loc[gdf2['gagnant']=="RN",['fill_coolor']] =   "#0000ff"
gdf2.loc[gdf2['gagnant']=="LREM",['fill_coolor']] = "#ff00ff"
### jointure avec les donnes geographiques ###


map_res = folium.Map(location=[45,1], zoom_start=4,tiles='cartodbpositron')

folium.GeoJson(
    gdf2,
    style_function = lambda feature: {
    'fillColor': feature['properties']['fill_coolor'],
    'weight' : 2,
    'color' : feature['properties']['fill_coolor']}
    ).add_to(map_res)

map_res.save('comp_RN_LREM.html')

### On cree une fonction qui retourne abs(xi-xj) ###

def indice_gini(x):
    """
    Cette fonction permet de calculer le coefficient de gini d une serie statistique
    """
    x = np.array(x)
    n = np.size(x) ## nombre d elements du vecteur 
    M = np.mean(x)
    G = 0
    ### On import le module itertools pour fair un produit cartesien ###
    import itertools
    for i in itertools.product(x,x):
        G = G + np.abs(i[0]-i[1])
    G = G / n**2
    return G/(2*M)
    
### On ajoute  l indice de Gini dans la table ####

data['gini'] = data[['LFI','PP','EEV','LREM','LR','RN']].apply(indice_gini,axis=1)

data['gini'].describe()

### Test des distribution ####



#### boxplot de la distribution ###

fig,ax = plt.subplots(figsize=(10,10))
ax = sns.boxplot(x="gagnant", y="gini", data=election)
plt.title("Boxplot des inidices de Gini sur la concentration de la repartition des votes")
plt.show()

stats.kruskal(data.loc[data['gagnant']=="RN",['gini']],
                data.loc[data['gagnant']=="LREM",['gini']])

stats.ttest_ind(data.loc[data['gagnant']=="RN",['gini']],
                data.loc[data['gagnant']=="LREM",['gini']])

stats.ttest_ind(data.loc[data['gagnant']=="RN",['gini']],
                data.loc[data['gagnant']=="LREM",['gini']],equal_var=False)

data[['code_dep','lib_dep','RN','LREM','gagnant','gini']].sort_values(by="gini",ascending=False).head(n=20)

data.groupby('gagnant')['gini'].aggregate(['count','min','mean','std','max'])


data.loc[data.code_dep.isin (['02','52','62']),['lib_dep','LFI','PP','EEV','LREM','LR','RN']]

data[['code_dep','lib_dep','RN','LREM','gagnant','gini']].sort_values(by="gini",ascending=False).tail(n=5)

data.loc[data.code_dep.isin (['48','46','87','19','93']),['lib_dep','LFI','PP','EEV','LREM','LR','RN']]



#####################################################################
############# Statistique multidimensionnel #############
#####################################################################
data[partis].describe()
matrice_corr = data[partis].corr()

fig,ax = plt.subplots(figsize=(9,9))
ax = sns.heatmap(matrice_corr,center=0,vmax=1, vmin=-1,cmap="YlGnBu")
plt.title("Correlations des votes des partis")
plt.show()

################" Standardisation des donnes #################
# On Cree une nouvelle table pour l'ACP
X = data[['code_dep']+partis]
### On change l'indexe afin de ne garder que des donnees numeriques
X = X.set_index('code_dep')
n,p = X.shape
#### On charge lobjet standardscar pour normaliser les donnes
from sklearn.preprocessing import StandardScaler
## Instanciation 
sc = StandardScaler()
#### On centre et on reduit les donnees ####
Z = sc.fit_transform(X)
print(np.mean(Z,axis=0))
print(np.std(Z,axis=0,ddof=0))


################ ACP  #################
## chargement de lobjet PCA ##
from sklearn.decomposition import PCA
acp = PCA(svd_solver='full')
#calculs : Recuperation des coordonnes factorielles ##
coord = acp.fit_transform(Z)
## vecteur des valeurs propres: On corrige car c'est la variance estimee qui est utilisee ####
vecteur_propres =  (n-1)/n*acp.explained_variance_
#proportion de variance expliquée
print(acp.explained_variance_ratio_)

###### Graphiques des valeurs propores ####
#scree plot
fig,ax = plt.subplots(figsize=(9,9))
plt.plot(np.arange(1,p+1),vecteur_propres)
plt.title("Eboulit des valeurs propres")
plt.ylabel("Valeur propres")
plt.xlabel("Axe factoriel")
plt.show()


#### Graphiques des departements #####
np.min(coord,axis=0)
np.max(coord,axis=0)

#################Graphique sur l'axe 1 et 2 ######################

fig,ax = plt.subplots(figsize=(11,11))
ax.set_xlim(-5.5,5.5) #même limites en abscisse
ax.set_ylim(-5.5,5.5) #et en ordonnée
## placement desetiquettes ####
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))

#ajouter les axes
plt.plot([-5.5,5.5],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-5.5,5.5],color='silver',linestyle='-',linewidth=1)
#affichage
plt.title("Projection sur le plan factoriel")
plt.ylabel("Axe factoriel 2")
plt.xlabel("Axe factoriel 1")
plt.show()

#################Graphique sur l'axe 1 et 3 ######################

fig,ax = plt.subplots(figsize=(11,11))
ax.set_xlim(-5.5,5.5) #même limites en abscisse
ax.set_ylim(-5.5,5.5) #et en ordonnée
## placement desetiquettes ####
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,2]))

#ajouter les axes
plt.plot([-5.5,5.5],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-5.5,5.5],color='silver',linestyle='-',linewidth=1)
#affichage
plt.title("Projection sur les plan 1 et 3")
plt.ylabel("Axe factoriel 3")
plt.xlabel("Axe factoriel 1")
plt.show()


### Contribution aux axes ###########

#contributions aux axes
ctr = coord**2
for j in range(p):
 ctr[:,j] = ctr[:,j]/(n*vecteur_propres[j])

ctr_fdf = pd.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]})

### Contribution des individus sur les plan principal #########
ctr_fdf.sort_values(by='CTR_1',ascending=False).head(n=20)
ctr_fdf.sort_values(by='CTR_2',ascending=False).head(n=10)

#################### Cercle des correlation #####################
#racine carrée des valeurs propres
sqrt_lambda = np.sqrt(vecteur_propres)

#corrélation des variables avec les axes
corvarfac = np.zeros((p,p))

for k in range(p):
    corvarfac[:,k] = acp.components_[k,:] * sqrt_lambda[k]
    
######### Cercle des correlation #############

#cercle des corrélations
fig, ax = plt.subplots(figsize=(9,9))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(X.columns[j],(corvarfac[j,0],corvarfac[j,1]))

#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_artist(cercle)
plt.title("Cercle des correlations")
#affichage
plt.show()

#### Qualite de representation des variables: cos_carree ##########
cos_carre = corvarfac**2
###Creation du dataframe sur les coscarre ###
cos_carre_df = pd.DataFrame({'id':X.columns,'COS2_1':cos_carre[:,0],'COS2_2':cos_carre[:,1]})

#### Contribution des variables ################
ctrvar = cos_carre
for k in range(p):
    ctrvar[:,k] = ctrvar[:,k]/vecteur_propres[k]

ctrvar = pd.DataFrame({'id':X.columns,'Contribution axe 1':ctrvar[:,0],'Contribution Axe 2':ctrvar[:,1]})


############Variable supplementaire################

taux_chomage = pd.read_excel("sl_cho_2018T4.xls",sheet_name="Département")

### Datamanagement sur le TF #############
taux_chomage = taux_chomage.iloc[:,[0,1,149]]
taux_chomage = taux_chomage.iloc[3:99,:]
taux_chomage.columns=['code_dep','lib_dep','T4_2018']
taux_chomage['T4_2018'] = pd.to_numeric(taux_chomage['T4_2018'])

########### Chargement des donnees de population #####

pop_age = pd.read_excel("TCRD_021.xls",sheet_name="DEP")

pop_age = pop_age.iloc[:,[0,1,5,6,7]]
pop_age = pop_age.iloc[3:99,:]
pop_age.columns=['code_dep','lib_dep','pop_0_24','pop_25_59','pop_plus_60']

####apply conversion numeric###
pop_age[['pop_0_24','pop_25_59','pop_plus_60']] = pop_age[['pop_0_24','pop_25_59','pop_plus_60']].apply(pd.to_numeric,axis=0)

#### Creation du dataframe finale ####

election_var_supp = pd.merge(left = taux_chomage,
                             right = pop_age,
                             how = 'inner',
                             left_on = 'code_dep',
                             right_on = 'code_dep')

election_var_supp.drop(columns = ['lib_dep_x','lib_dep_y'], inplace = True)


### Ajout des donnees sur la fiscalite ####

### On cree une fonction qui faire creer le string de l'onglet dep ###
def get_dep_sheet_ircom(dep):
    """
    Fonction qui ajoute un zeros sur les numeros de departements
    """
    return dep+'0'
        

list_dep_ircom = [get_dep_sheet_ircom(_) for _ in X.index.tolist()]

#### On cree maitenant une fonction qui recupere la valeur du taux d'imposition ####


taux_menage_impose = pd.DataFrame()

def get_taux_men_imp_ircom(dep):
    """
    Cette fonction recupere le taux de menage impose
    """
    x = pd.read_excel("ircom_2017_revenus_2016.xlsx",sheet_name=dep)
    nbr_foyer_fisc = x.iloc[14,5]
    nbr_foyer_fisc_imp = x.iloc[14,8]
    _temp = pd.DataFrame({'code_dep':dep[0:2],'taux_menage_imp':[nbr_foyer_fisc_imp/nbr_foyer_fisc]})
    return _temp
    
  
for dep_ircom in list_dep_ircom:
    taux_menage_impose = taux_menage_impose.append(get_taux_men_imp_ircom(dep_ircom))


election_var_supp = pd.merge(left = election_var_supp,
                             right = taux_menage_impose,
                             how = 'inner',
                             left_on = 'code_dep',
                             right_on = 'code_dep')


### Lecture des donnees des diplomes ####

nb_diplomes = pd.read_csv("fr_esr_sise_diplomes_delivres_esr_public.csv",sep=";",usecols = [2])

#### On cree la colonne departement en faisant un substring #####

nb_diplomes['code_dep'] = nb_diplomes['ETABLISSEMENT_code'].str[1:3]

nb_diplomes.drop(columns=['ETABLISSEMENT_code'],inplace = True)

nb_diplomes['dummy'] = 1

### On fait group by pour recuperer le noombre de diplome par departement ###

nb_diplomes = nb_diplomes.groupby('code_dep')['dummy'].agg(np.sum).reset_index()

nb_diplomes.rename(columns={'dummy':'nbr_diplome'},inplace = True)

nb_diplomes.plot(kind='hist',figsize=(9,9))

decile = [_/10 for _ in range(1,11)]

nb_diplomes.plot(kind='kde',figsize=(9,9))

nb_diplomes.describe(percentiles=decile)



#### Lecture des donnees de population ###

pop_2019_estimes = pd.read_excel("TCRD_004.xls",sheet_name="DEP",usecols = [0,1,2]).iloc[3:99,:]
pop_2019_estimes.columns = ['code_dep','lib_dep','pop']
pop_2019_estimes['pop'] = pd.to_numeric(pop_2019_estimes['pop'])


### jointure entre les donnees de population et les donnees de diplomes

diplome_rate_pop = pd.merge(left = pop_2019_estimes,
                           right = nb_diplomes,
                           left_on = 'code_dep',
                           right_on = 'code_dep',
                           how = 'left')

#### On remplace vide par 0 ####

diplome_rate_pop.fillna(0,inplace=True)

### ajout de la colonne diplome rate ####

diplome_rate_pop['diplome_rate'] = (diplome_rate_pop['nbr_diplome']*1000)/diplome_rate_pop['pop']

diplome_rate_pop['diplome_rate'].plot(kind='kde',figsize=(9,9))

diplome_rate_pop['diplome_rate'].describe(percentiles=decile)

def bining_diplome_rate(x):
    """
    Cree la classe pour le diplome
    """
    if x <= 0:
        return 'A-'
    elif x <= 2.45:
        return 'A'
    else:
        return 'A+'
    
diplome_rate_pop['diplome_rate_range'] =diplome_rate_pop['diplome_rate'].apply(bining_diplome_rate)

### Reordoring with X dataframe #########


election_var_supp = election_var_supp.set_index('code_dep')

election_var_supp = election_var_supp.ix[X.index.tolist()]


#corrélation avec les axes factoriels
corSupp = np.zeros((election_var_supp.shape[1],p))

for k in range(p):
    for j in range(election_var_supp.shape[1]):
        corSupp[j,k] = np.corrcoef(election_var_supp.iloc[:,j],coord[:,k])[0,1]


#cercle des corrélations avec les var. supp
fig, ax = plt.subplots(figsize=(9,9))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

#variables actives
for j in range(p):
    plt.annotate(X.columns[j],(corvarfac[j,0],corvarfac[j,1]))
    
#variables illustratives

for j in range(election_var_supp.shape[1]):
    plt.annotate(election_var_supp.columns[j],(corSupp[j,0],corSupp[j,1]),color='g')
    
#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_artist(cercle)

plt.title("Cercle des correlations + Var supplementaires")
#affichage
plt.show()


############ Graphiques des individus sur le plan principale ####

#modalités de la variable qualitative

### ON cree un dataframe avec les coordonees des composantes principales ###
coord_df = pd.DataFrame({'code_dep':X.index,'coord_1':coord[:,0],'coord_2':coord[:,1]})
### Jointure avec les donnees des diplomes #####
coord_diplome = pd.merge(left = coord_df,
                         right = diplome_rate_pop[['code_dep','diplome_rate_range']],
                         left_on = 'code_dep',
                         right_on = 'code_dep',
                         how = 'inner')

### creation du Barycentre des trois classes ######
coord_diplome_barycentre = coord_diplome.groupby('diplome_rate_range')['coord_1', 'coord_2'].aggregate(['mean']).reset_index()
###Changement des noms du dataframe ####
coord_diplome_barycentre.columns = ['diplome_rate_range','coord_1','coord_2']



### Graphique avec les departement en couleur #############
diplome_rate_pop = diplome_rate_pop[['code_dep','diplome_rate_range']].set_index('code_dep')

diplome_rate_pop = diplome_rate_pop.ix[X.index.tolist()]
modalites = ['A-','A','A+']
couleurs = ['red','orange','green']
#faire un graphique en coloriant les points
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-5.5,5.5) #même limites en abscisse
ax.set_ylim(-5.5,5.5) #et en ordonnée
#pour chaque modalité de la var. illustrative
for c in range(len(modalites)):
    #numéro des individus concernés
    numero = np.where(diplome_rate_pop == modalites[c])
    #les passer en revue pour affichage
    for i in numero[0]:
        plt.annotate(X.index[i],(coord[i,0],coord[i,1]),color=couleurs[c])
        
#ajouter les axes
plt.plot([-5.5,5.5],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-5.5,5.5],color='silver',linestyle='-',linewidth=1)
#affichage
plt.title("Projection des departement avec taux de diplomes")
plt.ylabel("Axe factoriel 1")
plt.xlabel("Axe factoriel 2")
plt.show()

##################"Graphiques avec les barycentres #######
fig,ax = plt.subplots(figsize=(11,11))
ax.set_xlim(-5.5,5.5) #même limites en abscisse
ax.set_ylim(-5.5,5.5) #et en ordonnée
## placement desetiquettes ####
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))

###placement des barycentres #############"
for c in range(len(modalites)):
    numero = np.where(coord_diplome_barycentre.diplome_rate_range == modalites[c])
    plt.annotate(modalites[c],(coord_diplome_barycentre.iloc[numero[0],1],coord_diplome_barycentre.iloc[numero[0],2]),color=couleurs[c],size=16,weight = 'bold')
#ajouter les axes
plt.plot([-5.5,5.5],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-5.5,5.5],color='silver',linestyle='-',linewidth=1)
#affichage
plt.title("Projection sur le plan factoriel avec classe du taux de diplomé")
plt.ylabel("Axe factoriel 2")
plt.xlabel("Axe factoriel 1")
plt.show()


###############""Carte de la France avec les departements coloriés selon les partis ###################

carte_parti_dep = election[['code_dep']]

carte_parti_dep['couleur'] = np.nan

#### Replissage pour lrem #####

carte_parti_dep.loc[carte_parti_dep.code_dep.isin (['75','92','35','44','78',
                                                    '94','69','29','64','74','49'])
,['couleur']] = '#800080'
    

#### Remplissage pour le FN ##########

carte_parti_dep.loc[carte_parti_dep.code_dep.isin (['02','62','52','08','80',
                                                    '10','60','55','70','66','89'])
,['couleur']] = '#000080'

    
#### Remplissage pour le LFI et PP ##########

carte_parti_dep.loc[carte_parti_dep.code_dep.isin (['09','93','11','24','65',
                                                    '87','40','46'])
,['couleur']] = '#ff00ff'
    
    
#### Remplissage pour LR ##########

carte_parti_dep.loc[carte_parti_dep.code_dep.isin (['06','2A','2B','43']),['couleur']] = '#0000ff'

carte_parti_dep.couleur.fillna('#808080',inplace=True)
                               
#### Carte finale ##########

                            
map_final = folium.Map(location=[45,1], zoom_start=4,tiles='cartodbpositron')


gdf3 = pd.merge(left = gdf,
                right = carte_parti_dep,
                left_on = 'code_insee',
                right_on = 'code_dep')

gdf3.drop(columns=['code_dep'],inplace = True)

folium.GeoJson(
    gdf3,
    style_function = lambda feature: {
    'fillColor': feature['properties']['couleur'],
    'weight' : 2,
    'color' : feature['properties']['couleur']}
    ).add_to(map_final)

map_final.save('carte_partis_couleur.html')