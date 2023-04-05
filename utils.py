import pandas as pd

def result_anova(colonneQualitative, dataframe, seuil = 0.05):
  from statsmodels.formula.api import ols
  from statsmodels.stats.anova import anova_lm

  dataframe = dataframe.dropna()
  # On récupère les noms des colonnes quantitatives
  listeColonnesQuantitatives = dataframe.select_dtypes(exclude=[object]).columns
  resultatAnova = []
  for colonne in listeColonnesQuantitatives:
    print(colonne)
    entrainement = ols(f"{colonne} ~ {colonneQualitative}", data=dataframe).fit()        
    st_p = anova_lm(entrainement).loc[colonneQualitative, "PR(>F)"]
    if st_p > seuil:
      resultatAnova.append([colonneQualitative, colonne, "Il n'y a pas de lien", st_p])
    elif st_p < seuil:
      resultatAnova.append([colonneQualitative, colonne, "Il y a un lien", st_p])
  return pd.DataFrame(resultatAnova, columns=[colonneQualitative, "Colonne(s) numérique(s)", "Résultat Anova", "P-value"])

def chi2(dataframe, colonne1, colonne2):
  from scipy.stats import chi2_contingency
  st_chi2, st_p, st_dof, st_exp  = chi2_contingency(dataframe[[colonne1, colonne2]].pivot_table(index=colonne1, columns=colonne2, aggfunc=len).fillna(0))
  if st_p > 0.05:
    print(f"Il n'y pas pas de lien entre les colonnes {colonne1} et {colonne2}")
  elif st_p < 0.05:
    print(f"Il y a un lien entre les colonnes {colonne1} et {colonne2}")

  print(f"La p-value vaut {st_p}\n")
  
def get_fill_ratio(pourcentage, df):
  serie = (df.notna().sum()*100/len(df)).sort_values()
  df = pd.DataFrame({"Pourcentage du taux de remplissage" :  serie})
  df = df[df["Pourcentage du taux de remplissage"] >= pourcentage]
  print(f"Nombre de colonne(s) remplie a {pourcentage}% :", len(df))
  for i in range(len(df)):
    print(f"{df.index[i]} : {df.values[i][0].round(2)}")
  return df.index