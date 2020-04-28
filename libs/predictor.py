import pandas as pd
import numpy as np
from datetime import datetime
from fbprophet import Prophet
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from utils.stdout_silencer import suppress_stdout_stderr
import plotly.graph_objs as go
import plotly.offline as py

import logging
logger = logging.getLogger("fbprophet")
logger.setLevel(logging.ERROR)


def run_model(ts, name, date=datetime.now().strftime("%Y-%m-%d"), **args):
    m = CovidModel(ts, name, date, **args)
    return m()


def run_wrapper(args):
    import logging
    logger = logging.getLogger("fbprophet")
    logger.setLevel(logging.ERROR)
    return run_model(**args)


class CovidModel:
    def __init__(
        self, ts, name="default", date=datetime.now().strftime("%Y-%m-%d"), **args
    ):
        # Model Essentials
        self.name = name
        self.date = date
        self.ts = ts
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.pred_df = pd.DataFrame()
        self.args = args
        # Config arguments
        self.validation_steps = args.get("validation_steps", 7)
        self.train_data_points = args.get("train_data_points", 0)
        self.steps_to_predict = args.get("steps_to_predict", 10)
        self.clip_to_zero = args.get("clip_to_zero", False)
        self.prophet_args = args.get("prophet_args", {})
        #
        self.model = Prophet(**self.prophet_args)

    def __call__(self):
        return self.run()

    def split_ts(self):
        df = self.ts.copy()
        if self.validation_steps and self.train_data_points:
            self.val_df = df.iloc[-self.validation_steps :]
            self.train_df = df.iloc[
                -(
                    self.train_data_points + self.validation_steps
                ) : -self.validation_steps
            ]
        elif self.train_data_points:
            self.train_df = df.iloc[-self.train_data_points :]
        elif self.validation_steps:
            self.val_df = df.iloc[-self.validation_steps :]
            self.train_df = df.iloc[: -self.validation_steps]
        else:
            self.train_df = df.copy()
        return self

    def fit(self):
        with suppress_stdout_stderr():
            self.model.fit(self.train_df)
        return self

    def forecast(self):
        self.future = self.model.make_future_dataframe(
            periods=len(self.ts) - len(self.train_df) + self.steps_to_predict
        )
        self.pred_df = self.model.predict(self.future)
        return self

    def run(self):
        return self.split_ts().fit().forecast()

    def plot(self):
        self.model.plot_components(self.pred_df)

        # Draw forecast results
        self.model.plot(self.pred_df, xlabel="Date", ylabel="Value")

        # Combine all graps in the same page
        plt.ylabel("Value")
        plt.show()


class CovidPredictor:
    def __init__(
        self,
        df,
        date=datetime.now().strftime("%Y-%m-%d"),
        kpi="new_hosp",
        time_key="jour",
        area_key="dep",
        is_hierarchical=False,
        multiprocess=False,
        **args,
    ):
        # Essentials
        self.init_df = df.copy()
        self.df = df.copy()
        self.date = date
        self.kpi = kpi
        self.time_key = time_key
        self.area_key = area_key
        self.is_hierarchical = is_hierarchical
        self.multiprocess = multiprocess

        # Config arguments
        self.args = args

        # Models Record
        self.model_arguments = []
        self.models = {}

    def __call__(self):
        self._generate_time_series()
        if self.multiprocess:
            self.parallel_model()
        else:
            self.serial_model()

    def _generate_time_series(self):
        areas = self.df[self.area_key].unique()
        self.model_arguments = [
            {
                **{
                    "ts": self.df.query(f"{self.area_key} == '{a}'")
                    .filter(items=[self.time_key, self.kpi])
                    .groupby(self.time_key)
                    .sum()
                    .reset_index()
                    .rename(columns={self.time_key: "ds", self.kpi: "y"}),
                    "name": f"{a}",
                    "args": self.args,
                },
                **self.args,
            }
            for a in areas
        ]
        if not self.is_hierarchical:
            self.model_arguments.append(
                {
                **{
                    "ts": self.df
                    .filter(items=[self.time_key, self.kpi])
                    .groupby(self.time_key)
                    .sum()
                    .reset_index()
                    .rename(columns={self.time_key: "ds", self.kpi: "y"}),
                    "name": "00",
                    "args": self.args,
                },
                **self.args,
            }
            )
        return self

    def serial_model(self):
        start_time = time.time()
        self.results = list(
            map(lambda args: run_model(**args), tqdm(self.model_arguments))
        )
        print("--- %s seconds ---" % (time.time() - start_time))
        return self

    def parallel_model(self):
        if len(logger.handlers) == 1:
            logger.setLevel(logging.ERROR)
        start_time = time.time()
        p = Pool(cpu_count())
        self.results = list(
            tqdm(
                p.imap(run_wrapper, self.model_arguments),
                total=len(self.model_arguments),
            )
        )
        p.close()
        p.join()
        print("--- %s seconds ---" % (time.time() - start_time))
        return self

    def get_results(self):
        return pd.concat([r.pred_df.assign(area=r.name) for r in self.results], axis=0)


class CovidMetric:
    def __init__(self, predictor: CovidPredictor):
        self.predictor = predictor
        self._recolt()

    def _recolt(self):
        self.eval = pd.concat(
            [
                r.val_df.merge(
                    r.pred_df.filter(["ds", "yhat"]), on="ds", how="inner"
                ).assign(name=r.name)
                for r in self.predictor.results
            ]
        )
        self.eval = self.eval.assign(yhat=self.eval.yhat.clip(lower=0))
        return self

    @property
    def mape(self):
        return abs((self.eval.y - self.eval.yhat) / (self.eval.y)).replace([np.inf, -np.inf], np.nan).dropna().mean()

    @property
    def oom(self):
        return (
            (np.log(self.eval.yhat + 1) / np.log(self.eval.y + 1))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .mean()
        )

    @property
    def oomerror(self):
        return (
            (
                (np.log(self.eval.yhat + 1) - np.log(self.eval.y + 1))
                / np.log(self.eval.y + 1)
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .mean()
        )

    def _plotly(self, X, Y, title, xaxis, yaxis):
        abs((self.eval.y - self.eval.yhat) / (self.eval.yhat))
        trace = go.Scatter(
            name="Actual Value",
            mode="markers",
            x=X,
            y=Y,
            marker=dict(color="red", line=dict(width=1)),
        )
        data = [trace]
        layout = dict(
            title=title,
            xaxis=dict(title=xaxis, ticklen=2, zeroline=True),
            yaxis=dict(title=yaxis, ticklen=2, zeroline=True),
        )
        figure = dict(data=data, layout=layout)
        py.offline.iplot(figure)

    def pred_vs_gt(self):
        self._plotly(
            X=list(self.eval.y),
            Y=list(self.eval.yhat),
            title="Prediction VS Ground Truth",
            xaxis="Ground Truth",
            yaxis="Prediction",
        )

    def mape_vs_gt(self):
        self._plotly(
            X=list(self.eval.y),
            Y=list(abs((self.eval.y - self.eval.yhat) / (self.eval.y)).replace([np.inf, -np.inf], np.nan).dropna()),
            title="MAPE VS Ground Truth",
            xaxis="Ground Truth",
            yaxis="MAPE",
        )
    
    def oom_vs_oomerror(self):
        self._plotly(
            X=list(((np.log(self.eval.yhat + 1) - np.log(self.eval.y + 1))/ np.log(self.eval.y + 1)).replace([np.inf, -np.inf], np.nan)),
            Y=list((np.log(self.eval.yhat + 1) / np.log(self.eval.y + 1)).replace([np.inf, -np.inf], np.nan)),
            title="OOM VS OOM Error",
            xaxis="OOM Error",
            yaxis="OOM",
        )
    
    def oom_vs_mape(self):
        self._plotly(
            X=list(abs((self.eval.y - self.eval.yhat) / (self.eval.y)).replace([np.inf, -np.inf], np.nan).dropna()),
            Y=list((np.log(self.eval.yhat + 1) / np.log(self.eval.y + 1)).replace([np.inf, -np.inf], np.nan)),
            title="OOM VS MAPE",
            xaxis="MAPE",
            yaxis="OOM",
        )


