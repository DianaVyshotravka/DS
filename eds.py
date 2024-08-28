import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, classification_report, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from math import log1p


class MonthExtractor(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.month)).to_numpy(), (-1, 1))

    def transform(self, X):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.month)).to_numpy(), (-1, 1))

    def fit(self, X, y=None):
        return self


class HourExtractor(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.hour)).to_numpy(), (-1, 1))

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.hour)).to_numpy(), (-1, 1))


class DayExtractor(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.day)).to_numpy(), (-1, 1))

    def transform(self, X):
        return np.reshape((pd.to_datetime(X[:, 0], format='mixed').map(lambda x: x.day)).to_numpy(), (-1, 1))

    def fit(self, X, y=None):
        return self



def histplot(df, hue, **kwargs):
    columns = df.columns
    num_rows = (len(columns))-1
    fig, axes = plt.subplots(nrows=num_rows, figsize=(20, num_rows * 5))
    for i, column in enumerate(columns):
        row = i
        if column == hue:
            axes[row].set_axis_off()
            continue
        else:
            sns.histplot(df, x = column, hue = hue, ax = axes[row], **kwargs)
            for k in range(len(axes[row].containers)):
                axes[row].bar_label(axes[row].containers[k], fontsize=10)
                axes[row].set_title(f'Visualization of {column}')
    fig.tight_layout()
    return fig, axes

def histplot_simple(df, **kwargs):
    columns = df.columns
    num_rows = (len(columns))-1
    fig, axes = plt.subplots(nrows=num_rows, figsize=(20, num_rows * 5))
    for i, column in enumerate(columns):
        row = i
        sns.histplot(df, x = column,  ax = axes[row], **kwargs)
        for k in range(len(axes[row].containers)):
            axes[row].bar_label(axes[row].containers[k], fontsize=10)
            axes[row].set_title(f'Visualization of {column}')
    fig.tight_layout()
    return fig, axes



def visualize(df, funk, **kwargs):
    columns = df.columns
    num_rows = (len(columns))
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(15, num_rows * 5))
    for i, column in enumerate(columns):
        funk(df[column], ax=i, **kwargs)
        axes[i].set_title(f'Visualization of {column}')
    plt.tight_layout()
    plt.show()


def catplot_visualize(df, **kwargs):
    columns = df.columns
    num_rows = (len(columns))-1
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 5))
    for i, column in enumerate(columns):
        row = i
        sns.histplot(df,x = column, ax=axes[row, 0], **kwargs)
        sns.boxplot(df, x=column,  ax=axes[row, 1], **kwargs)
        axes[row].set_title(f'Visualization of {column}')
    plt.tight_layout()
    plt.show()


def categorical2d(df, row, column, cmap='YlGn', square=True, fs = (10,5)):
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
    sns.heatmap(pd.crosstab(df[row], df[column]), annot=True, cbar=False, cmap=cmap, fmt='d', ax=axs[0], square=square)
    sns.heatmap(pd.crosstab(df[row], df[column], normalize='index'), annot=True, cbar=False, cmap=cmap, fmt='.2f',
                ax=axs[1], square=square)
    sns.heatmap(pd.crosstab(df[row], df[column], normalize='columns'), annot=True, cbar=False, cmap=cmap, fmt='.2f',
                ax=axs[2], square=square)

    axs[0].set_title('Counts', fontsize=10)
    axs[1].set_title('Counts normalized by rows', fontsize=10)
    axs[2].set_title('Counts normalized by columns', fontsize=10 )

    axs[1].set(ylabel='')
    axs[2].set(ylabel='')

    axs[0].set(xlabel='')
    axs[2].set(xlabel='')

    fig.tight_layout()
    return fig, axs


def conf_matrix(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            colorbar=False, ax=axes[0])
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize= 'true',
                                            colorbar=False, ax = axes[1])
    axes[1].set_title('Normalized by recall')
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize= 'pred',
                                            colorbar=False, ax = axes[2])
    axes[2].set_title('Normalized by precision')
    plt.tight_layout()
    return fig, axes


def categ_visualize(df, **kwargs):
    columns = df.columns
    num_rows = (len(columns) - 1) // 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 5))
    for i, column in enumerate(columns):
        row = i // 2
        col = i % 2
        sns.histplot(df, x = column, ax = axes[row, col], **kwargs)
        axes[row, col].set_title(f'Visualization of {column}')
    plt.tight_layout()
    plt.show()


def numerical_visualize(df, **kwargs):
    columns = df.columns
    num_rows = (len(columns))
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 5))
    for i, column in enumerate(columns):
        row = i
        sns.histplot(df, x=column, ax=axes[row, 0], kde=True, **kwargs)
        axes[row, 0].set_title(f'Visualization of {column}')
        sns.boxplot(df, x=column, ax=axes[row, 1], **kwargs)
        axes[row, 0].set_title(f'Visualization of {column}')
    plt.tight_layout()
    plt.show()

def regression_report(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray,
                          x_train: pd.Series | np.ndarray) -> dict[str, float]:
    return ({
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R^2': r2_score(y_true, y_pred),
        'Adjusted R^2': 1 - (1 - r2_score(y_true, y_pred)) * (x_train.shape[0] - 1) / (
                        x_train.shape[0] - x_train.shape[1] - 1)
    })


def report(x_train, x_test, y_test_raw, model, target_transformer=None):

    if target_transformer is None:
        return pd.DataFrame(
            [
                regression_report(y_test_raw, np.reshape(model.predict(x_test), (-1, 1)), x_train),
            ],
            index=['Test set']
        ).T
    else:
        return pd.DataFrame(
            [
                regression_report(y_test_raw,
                                  target_transformer.inverse_transform(np.reshape(model.predict(x_test), (-1, 1))), x_train),
            ],
            index=['Test set']
        ).T



class TransformerList:
    def __init__(self, transformers):
        self.transformers = transformers

    def fit_transform(self, data, y=None):
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        return data

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data
    def inverse_transform(self, data):
        for transformer in self.transformers[::-1]:
            data = transformer.inverse_transform(data)
        return data


class StandartLog:
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler(*args, **kwargs)

    def fit_transform(self, data):
        return self.scaler.fit_transform(np.vectorize(lambda x: log1p(1 / x))(data))

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return self.scaler.transform(np.vectorize(lambda x: log1p(1 / x))(data))
