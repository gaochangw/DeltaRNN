import warnings
import pandas as pd


class PandasLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []

    def load_log(self, log_stat):
        # Create Log List
        list_headers = []
        list_values = []
        for k, v in log_stat.items():
            list_headers.append(k)
            list_values.append(v)
        self.list_header = list_headers
        row = {}
        for header, value in zip(list_headers, list_values):
            row[header] = value
        self.loglist = [row]

    def write_log(self, append=True, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            try:
                df_pre = pd.read_csv(self.logfile)
                csv_founded = 1
            except FileNotFoundError:
                df_pre = None
                csv_founded = 0
            df = pd.DataFrame(self.loglist, columns=self.list_header)

            if csv_founded and append:
                # df = df_pre.append(df)
                df = pd.concat([df_pre, df])

            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)
