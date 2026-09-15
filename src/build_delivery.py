"""Reproduire les compléments du brief et préparer les artefacts de démonstration."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, export_text
from src.train import (DATA_PATH, RESULTS_DIR, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE,
    load_data, clean_data, split_features_target, build_preprocessor, optimize_models,
    evaluate_model, save_results)
from src.visualize import export_visuals, feature_label, new_figure, finish_figure, style_axis
from src.dashboard import build_dashboard
from src.inference import make_bundle, predict_clients
from src.monitor import evaluate_observed


def analyze_tree(pipeline, X_train, y_train, X_test, y_test, directory):
    """Exporter l'arbre réel, ses règles et sa complexité après optimisation."""
    tree=pipeline.named_steps['model']
    prep=pipeline.named_steps['prep']
    names=prep.get_feature_names_out()
    labels=[feature_label(name) for name in names]
    fig,ax=plt.subplots(figsize=(18,8))
    plot_tree(tree,feature_names=labels,class_names=['Reste','Part'],max_depth=2,filled=True,rounded=True,fontsize=9,ax=ax)
    ax.set_title('Arbre optimisé : les trois premiers niveaux (le modèle complet va jusqu’à la profondeur 5)',fontsize=17,pad=18)
    fig.tight_layout()
    fig.savefig(directory/'05_arbre_decision.png',dpi=160)
    fig.savefig(directory/'05_arbre_decision.pdf')
    plt.close(fig)
    rules=export_text(tree,feature_names=list(names),decimals=3,max_depth=10)
    (directory/'05_regles_arbre.txt').write_text('Seuils numériques exprimés dans les variables standardisées du pipeline.\nLes colonnes one-hot valent 0 ou 1. Classes pondérées à l’entraînement.\n\n'+rules)
    # Transformer la coupure racine en unité métier pour l'expliquer sans ambiguïté.
    feature_index=tree.tree_.feature[0]
    feature_name=names[feature_index]
    threshold=float(tree.tree_.threshold[0])
    root_split={'feature':feature_name,'threshold_transformed':threshold}
    if feature_name.startswith('num__'):
        num_cols=prep.transformers_[0][2]
        j=num_cols.index(feature_name.split('__',1)[1])
        scaler=prep.named_transformers_['num'].named_steps['scaler']
        root_split['threshold_original_units']=float(threshold*scaler.scale_[j]+scaler.mean_[j])
    stats={'depth':int(tree.get_depth()),'leaves':int(tree.get_n_leaves()),'nodes':int(tree.tree_.node_count),
        'root_split':root_split,'train_accuracy':accuracy_score(y_train,pipeline.predict(X_train)),
        'test_accuracy':accuracy_score(y_test,pipeline.predict(X_test)),
        'train_roc_auc':roc_auc_score(y_train,pipeline.predict_proba(X_train)[:,1]),
        'test_roc_auc':roc_auc_score(y_test,pipeline.predict_proba(X_test)[:,1])}
    # Retrouver les conditions de chaque feuille et remettre les seuils numériques en unités brutes.
    parents = {}
    for parent in range(tree.tree_.node_count):
        for child, side in [(tree.tree_.children_left[parent], '<='), (tree.tree_.children_right[parent], '>')]:
            if child != -1:
                parents[child] = (parent, side)
    def conditions(node):
        result = []
        while node in parents:
            parent, side = parents[node]
            name = names[tree.tree_.feature[parent]]
            threshold = float(tree.tree_.threshold[parent])
            if name.startswith('num__'):
                column = name.split('__', 1)[1]
                position = prep.transformers_[0][2].index(column)
                scaler = prep.named_transformers_['num'].named_steps['scaler']
                raw_threshold = threshold * scaler.scale_[position] + scaler.mean_[position]
                result.append(f'{column} {side} {raw_threshold:.3f}')
            else:
                modality = feature_label(name)
                result.append(f'{modality} : ' + ('modalité absente' if side == '<=' else 'modalité présente'))
            node = parent
        return list(reversed(result))
    # Décrire trois feuilles réellement atteintes, avec leurs effectifs train non pondérés.
    leaves=tree.apply(prep.transform(X_train))
    summaries=[]
    for node in pd.Series(leaves).value_counts().head(3).index:
        mask=leaves==node
        summaries.append({'leaf_id':int(node),'conditions':conditions(node),'train_clients':int(mask.sum()),'observed_churn_rate_train':float(np.asarray(y_train)[mask].mean())})
    stats['largest_leaves']=summaries
    (directory/'05_complexite_arbre.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    return stats


def analyze_forest(pipeline,X_test,y_test,directory):
    """Permuter chaque variable d'origine sur le test pour mesurer la baisse de ROC-AUC."""
    importance=permutation_importance(pipeline,X_test,y_test,scoring='roc_auc',n_repeats=10,random_state=RANDOM_STATE,n_jobs=1)
    result=pd.DataFrame({'variable':X_test.columns,'mean_auc_drop':importance.importances_mean,'std_auc_drop':importance.importances_std}).sort_values('mean_auc_drop',ascending=False)
    result.to_csv(directory/'06_importance_random_forest.csv',index=False)
    selected=result.head(12).sort_values('mean_auc_drop')
    fig=new_figure('Quelles variables aident la Random Forest ?', 'Importance par permutation sur le test | 10 répétitions | Baisse de ROC-AUC')
    ax=fig.add_axes([.26,.23,.66,.55]);style_axis(ax)
    ax.barh(selected.variable,selected.mean_auc_drop,xerr=selected.std_auc_drop,color='#2463a5',capsize=3)
    ax.axvline(0,color='#718096',linewidth=1)
    ax.set_xlabel('Baisse moyenne de ROC-AUC quand la variable est mélangée',fontsize=11)
    finish_figure(fig,directory,'06_importance_random_forest','Barres d’erreur : écart-type des permutations, pas intervalle de confiance.\nLes variables corrélées peuvent se substituer ; cette analyse descriptive ne prouve pas une cause.')
    return result


def compare_regression(X_train,y_train,X_test,y_test,directory):
    """Comparer une régression linéaire sur la cible 0/1, à titre pédagogique uniquement."""
    model=Pipeline([('prep',build_preprocessor(X_train)),('model',LinearRegression())]).fit(X_train,y_train)
    score=model.predict(X_test)
    pred=score>=.5
    result={'model':'Régression linéaire sur cible binaire (comparaison pédagogique)',
        'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),
        'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),
        'roc_auc':roc_auc_score(y_test,score),'rmse':float(np.sqrt(mean_squared_error(y_test,score))),
        'score_min':float(score.min()),'score_max':float(score.max()),
        'outside_0_1':int(((score<0)|(score>1)).sum()),'threshold':.5,
        'note':'Scores non bornés, non interprétables comme des probabilités. Comparaison distincte des trois classifieurs pondérés ; aucune optimisation sur le test.'}
    (directory/'comparaison_regression_lineaire.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    raw=load_data();df=clean_data(raw);X,y=split_features_target(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    print('Optimisation commune des trois modèles...',flush=True)
    models,searches=optimize_models(X_train,y_train)
    scores=[evaluate_model(name,model,X_test,y_test) for name,model in models.items()]
    save_results(scores,searches)
    baseline=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
    baseline_scores=evaluate_model('Référence naïve',baseline,X_test,y_test)
    export_visuals(y,X_test,y_test,models,scores+[baseline_scores])
    directory=RESULTS_DIR/'figures'
    print('Analyse de l’arbre et de la forêt...',flush=True)
    tree_stats=analyze_tree(models['Arbre de décision'],X_train,y_train,X_test,y_test,directory)
    importance=analyze_forest(models['Random Forest'],X_test,y_test,directory)
    compare_regression(X_train,y_train,X_test,y_test,RESULTS_DIR)
    build_dashboard(X_test,y_test,models,RESULTS_DIR/'dashboard')
    version='telco-logistique-'+hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12]+'-v1'
    bundle=make_bundle(models['Régression logistique'],X_train,version)
    (PROJECT_ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(bundle,PROJECT_ROOT/'models/churn.joblib')
    # Exemples tirés du test : ils démontrent le fonctionnement, pas une nouvelle validation.
    demo=df.loc[X_test.index[:20]].drop(columns='Churn').copy()
    examples=PROJECT_ROOT/'data/examples';examples.mkdir(parents=True,exist_ok=True)
    original_ids=demo.customerID.copy()
    demo['customerID']=[f'DEMO-{i:03d}' for i in range(1,len(demo)+1)]
    demo.to_csv(examples/'clients_demo.csv',index=False)
    outcomes=pd.DataFrame({'customerID':demo.customerID,'Churn':df.loc[demo.index,'Churn']})
    outcomes.to_csv(examples/'depart_observe_demo.csv',index=False)
    records=json.loads(demo.replace({np.nan:None}).to_json(orient='records'))
    (examples/'requete_api.json').write_text(json.dumps({'clients':records,'threshold':.5},ensure_ascii=False,indent=2)+'\n')
    demo_predictions,report=predict_clients(demo,bundle)
    monitoring_dir=RESULTS_DIR/'monitoring';monitoring_dir.mkdir(exist_ok=True)
    demo_predictions.to_csv(monitoring_dir/'predictions_demo.csv',index=False)
    (monitoring_dir/'lot_normal.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    changed=demo.copy();changed['Contract']='Offre inconnue';changed['TotalCharges']=''
    _,altered_report=predict_clients(changed,bundle)
    altered_report['scenario']='Données volontairement altérées pour tester les alertes ; ce n’est pas une dérive observée en production.'
    (monitoring_dir/'lot_altere.json').write_text(json.dumps(altered_report,ensure_ascii=False,indent=2)+'\n')
    performance=evaluate_observed(demo_predictions,outcomes)
    performance['scenario']='Démonstration sur 20 clients du test historique, pas des résultats futurs.'
    (monitoring_dir/'performance_demo.json').write_text(json.dumps(performance,ensure_ascii=False,indent=2)+'\n')
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'dataset':DATA_PATH.name,
        'dataset_sha256':hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),'raw_rows':len(raw),'clean_rows':len(df),
        'train_rows':len(X_train),'test_rows':len(X_test),'random_state':RANDOM_STATE,'test_size':TEST_SIZE,'cv':5,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__},
        'searches':searches,'model_version':version}
    (RESULTS_DIR/'visualisation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (PROJECT_ROOT/'models/model_card.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Dashboard, explications, modèle sauvegardé et rapports de suivi créés.',flush=True)


if __name__=='__main__':
    main()
