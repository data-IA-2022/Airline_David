import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import scikitplot as skplt
import time
import numpy as np 
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

def result_anova(colonneQualitative, dataframe, seuil = 0.05):
  from statsmodels.formula.api import ols
  from statsmodels.stats.anova import anova_lm

  dataframe = dataframe.dropna()
  # On récupère les noms des colonnes quantitatives
  listeColonnesQuantitatives = dataframe.select_dtypes(exclude=[object]).columns
  resultatAnova = []
  for colonne in listeColonnesQuantitatives:
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

def clean_columns(df):
  df.columns = [col.replace(' ', '_') for col in df.columns]
  df.columns = [col.replace('/', '_') for col in df.columns]
  df.columns = [col.replace('-', '_') for col in df.columns]
  return df


def get_rapport_clf(list_grid, X_train, X_test, y_train, y_test):
  labels = ['KNN', 'Gradient Boosting', 'Random Forest', 'Extreme Boosting']
  for clf, label in zip(list_grid, labels):
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    y_pred = clf.predict(X_test)
    print(label)
    print(f"time of fit : {np.round(end_time-start_time, 2)} seconds\n")
    print(f"Best Parameters : {clf.best_params_}")
    print(classification_report(y_test, y_pred, digits=4))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True,title=label)
    plt.show()


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()