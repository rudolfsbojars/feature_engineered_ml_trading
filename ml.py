import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, log_loss, 
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
import joblib
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV


# takes shape of "open", "high", "low", "close", "volume", "rsi", "ema20", "ema50", "bos_bull", "bos_bear", "fvg_up_active", "fvg_down_active", "poc", "va_high", "va_low", "poc_delta", "poc_volume"

FEATURE_SETS = {
    "base": ["log_return", "log_range", "volume"],
    "base_plus_rsi_and_ema": ["log_return", "log_range", "volume", "rsi", "ema20", "ema50"],
    "full": ["log_return", "log_range", "volume", "rsi", "ema20", "ema50", "bos_bull", "bos_bear", "fvg_up_active", "fvg_down_active", "poc", "va_high", "va_low", "poc_delta", "poc_volume"],
    "random": ["log_return", "log_range", "random_rsi", "random_binary_1", "random_binary_2"]
}

class MLM:
    def __init__(self, read_file,  save_file_name):
        self.save_file_name = save_file_name
        self.read_file = read_file 

    
    def run_all(self,  start_date=None, end_date=None):
        print("MLM: Running Sequence")
        
        folder_name = f"{start_date}_{end_date}".replace(":", "-")
        base_path = os.path.join(self.save_file_name, folder_name)

        os.makedirs(base_path, exist_ok=True)

        self.df = self.load_data(self.read_file, start_date, end_date)
        self.df = self.convert_to_difference_features(self.df)
        self.df = self.add_random_features(self.df)
        
        self.df = self.df.dropna().reset_index(drop=True)
        
        train, test = self.split_data(self.df, 0.8)
        
        train = self.label_data(train)
        test = self.label_data(test)
        
        print("IC and MI result: ", self.calculate_ic_mi(train))
        
        print("Train Data: ", train.head())
        print("Test Data: ", train.head())
        
        train, test, scalers = self.normalize_all(train, test)

        results = {}

        for name, features in FEATURE_SETS.items():

            X_train = train[features]
            X_test  = test[features]
            
            print(f"Model :{name}, \nTrain Data: {X_train.head(10)}")

            y_train = train["label"]
            y_test  = test["label"]

            model = self.train_xgb(X_train, y_train)

            metrics = self.evaluate(model, X_test, y_test, name)

            self.print_feature_importance(model, features)

            self.save_model(
                model,
                scalers,
                features,  
                path=os.path.join(base_path, f"{name}.pkl"),
                metrics=metrics,
            )

            results[name] = metrics
            
            with open(os.path.join(base_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)


        print("FINAL RESULTS:")
        for k, v in results.items():
            print(k, v)
            
    def load_data(self, file_name=None, start_date=None, end_date=None):
        print("MLM: Loading File")
        
        self.file_name = file_name
        df_local = pd.read_csv(file_name, parse_dates=["timestamp"])
        df_local = df_local.sort_values("timestamp").reset_index(drop=True)
        
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            df_local = df_local[df_local["timestamp"] >= start_date]

        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            df_local = df_local[df_local["timestamp"] <= end_date]
            
        return df_local
            
    def convert_to_difference_features(self, df):
        print("MLM: Getting Logs and Diffs")
        
        df_local = df.copy()

        df_local["log_return"] = np.log(df_local["close"] / df_local["close"].shift(1))

        df_local["log_range"] = np.log(df_local["high"] / df_local["low"])


        df_local["dist_to_poc"] = df_local["close"] - df_local["poc"]
        df_local["dist_to_vah"] = df_local["close"] - df_local["va_high"]
        df_local["dist_to_val"] = df_local["close"] - df_local["va_low"]

        df_local = df_local.dropna().reset_index(drop=True)

        return df_local
        
    def split_data(self, df, split_ratio=0.8, split_by_date=None):
        print("MLM: Splitting Data")

        df_local = df.copy()

        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
        df_local = df_local.sort_values("timestamp").reset_index(drop=True)

        if split_by_date is not None:

            split_by_date = pd.to_datetime(split_by_date)

            train = df_local[df_local["timestamp"] < split_by_date].copy()
            test  = df_local[df_local["timestamp"] >= split_by_date].copy()

        else:
            split_idx = int(len(df_local) * split_ratio)

            train = df_local.iloc[:split_idx].copy()
            test  = df_local.iloc[split_idx:].copy()

        if len(train) == 0 or len(test) == 0:
            raise ValueError("Invalid split: empty train or test set")

        return train, test
    
    def label_data(self, df, n_candles=10, tp_pct=0.01, sl_pct=0.01):
        print("MLM: Labeling Data")

        df_local = df.copy()

        close = df_local['close'].values
        high  = df_local['high'].values
        low   = df_local['low'].values

        labels = np.full(len(df_local), np.nan)

        for i in range(len(close)):

            entry = close[i]
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)

            label = np.nan

            for j in range(1, n_candles + 1):

                if i + j >= len(close):
                    break

                if high[i + j] >= tp:
                    label = 1
                    break
                if low[i + j] <= sl:
                    label = 0
                    break


            labels[i] = label

        df_local["label"] = labels
        df_local = df_local.dropna(subset=["label"])
        df_local["label"] = df_local["label"].astype(int)

        return df_local
    
    def normalize_all(self, df_train, df_test):
        print("MLM: Group-wise normalization")

        df_train = df_train.copy().dropna()
        df_test  = df_test.copy().dropna()

        group_returns  = ["log_return", "log_range"]
        group_spatial  = ["dist_to_poc", "dist_to_vah", "dist_to_val", "poc_delta"]
        group_activity = ["volume", "poc_volume"]
        group_trend    = ["ema20", "ema50"]

        scalers = {
            "returns": StandardScaler(),
            "spatial": StandardScaler(),
            "activity": StandardScaler(),
            "trend": StandardScaler(),
        }

        def apply(cols, scaler):
            cols = [c for c in cols if c in df_train.columns]
            if not cols:
                return

            scaler.fit(df_train[cols])

            df_train[cols] = scaler.transform(df_train[cols])
            df_test[cols]  = scaler.transform(df_test[cols])

        apply(group_returns, scalers["returns"])
        apply(group_spatial, scalers["spatial"])
        apply(group_activity, scalers["activity"])
        apply(group_trend, scalers["trend"])

        return df_train, df_test, scalers
    
    def add_random_features(self, df):
        print("MLM: Adding Random Features")
        
        df_local = df.copy()
        
        np.random.seed(1)

        df_local["random_rsi"] = np.random.uniform(0, 100, len(df_local))

        df_local["random_binary_1"] = np.random.randint(0, 2, len(df_local))
        df_local["random_binary_2"] = np.random.randint(0, 2, len(df_local))
        
        return df_local
    
    def optimize_xgb(self, X_train, y_train):
        print("MLM: Starting Grid Search for best parameters...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }

        xgb = XGBClassifier(random_state=1, eval_metric='logloss')

        grid_search = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid, 
            cv=3, 
            scoring='accuracy', # Meklē labāko Accuracy bakalaura piemini
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best accuracy: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_
    
    def train_xgb(self, X_train, y_train):
        print("MLM: Training XGBoost")

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.01,
            subsample=1,
            colsample_bytree=0.8,
            random_state=1,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        return model
    
    def evaluate(self, model, X_test, y_test, model_name="Model"):
        print(f"MLM: Evaluating {model_name}")

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        loss = log_loss(y_test, probs)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"Log Loss:  {loss:.4f}")

        """
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        fpr, tpr, _ = roc_curve(y_test, probs)
        ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='darkorange', lw=2)
        ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[0].set_title(f'ROC Curve - {model_name}')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].legend(loc="lower right")
        ax[0].grid(alpha=0.3)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        ax[1].plot(recall_vals, precision_vals, color='blue', lw=2)
        ax[1].set_title(f'Precision-Recall Curve - {model_name}')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        """

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": loss
        }
        
    def calculate_ic_mi(self, df):
        print("MLM: Calculating IC and MI")
        
        features = FEATURE_SETS["full"]
        
        X = df[features].dropna()
        y = df.loc[X.index, "label"]
        
        mi = mutual_info_classif(X, y, random_state=1)
        
        ic_results = []
        for col in features:
            corr, pval = spearmanr(X[col], y)
            ic_results.append({"feature": col, "IC": round(corr, 4), "p_value": round(pval, 4), "MI": round(mi[features.index(col)], 4)})
        
        result_df = pd.DataFrame(ic_results).sort_values("MI", ascending=False)
        print(result_df.to_string(index=False))
        
        result_df.to_csv(os.path.join(self.save_file_name, "ic_mi.csv"), index=False)
        
        return result_df
    
    def label_data_next_candle(self, df):
        print("MLM: Labeling Data (next candle) for MI and IC")

        df_local = df.copy()

        df_local["label"] = (df_local["close"].shift(-1) > df_local["close"]).astype(int)
        df_local = df_local.dropna(subset=["label"])

        return df_local
            
    def print_feature_importance(self, model, feature_columns):
        print("MLM: Feature Importance:")

        importances = model.feature_importances_

        for name, imp in sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp:.4f}")         
    
    def save_model(self, model, scalers, feature_columns, path="models/xgb_model.pkl", metrics=None, name=None):
        print("MLM: Saving model")

        os.makedirs("models", exist_ok=True)

        package = {
            "model": model,
            "scalers": scalers,
            "features": feature_columns,
            "metrics": metrics,
            "model_name": name,
            "created_at": str(datetime.now())
        }

        joblib.dump(package, path)

        print(f"MLM: Model saved to {path}")


if __name__ == '__main__':
    model = MLM("data/feature_extracted/15M/BTCUSDT-15m-2017-08-2026-03-features.csv", "models/BTC_15M/") #VARIABLE
    model.run_all(start_date="2018-10-01", end_date="2020-01-01") #VARIABLE