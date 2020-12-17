import pandas as pd


def pd_drive(key, **kwargs):
    url = f"https://drive.google.com/uc?export=download&id={key}"
    return pd.read_csv(url, **kwargs)

class Data:
    def ten_days(self):
        # Public google drive download keys:
        self.connections = "1_u1qHVNElC3aeFcY7UKBXlUBXxJBig9Y"
        self.data = "1obwJfVNACfIXhDcasVVcS8N6yiHLNrTI"
        self.embeddings = "1X-GshQBzXO9HgPi_4R0VuVtsR-SXtrMM"
        self.avg_esg = "1FIB_zV0xpZNAEiE-HbBD4GPcndU3aZqK"
        self.daily_esg = "1VJhHBieTX-SbLl76EcJSp40YQOJHBaPi"
        self.e_score = "1D9l9YZnRZNY4fTt-MjnQkFsVF2DsCSMJ"
        self.s_score = "1Z1WVo7LF1DtbDdwsXZnrqwPovJIdKjrY"
        self.g_score = "13ChPglmN1FKLkHErPICD7GcsSCeyDDkM"

    def one_month(self):
        self.connections = "1PqPHIw7-4qC7lBxNZWGtXQmhidHgwtNa"
        self.data = "1EX4Uf1-JiXKEGrwx6Pe9Dc6IJ35kD-D_"
        self.embeddings = "11h4A3quq90DT_JIoUzZx70VpZO-CBnX2"
        self.avg_esg = "1ZfIizZH7IKwXtX-MaqZJlr7yz0kTnmMs"
        self.daily_esg = "1Skip5ZNaoZemObSTRaG9Dl_AszwArcfA"
        self.e_score = "1QeL87KIUM5H1QiJe5Hgw2hXtrgszgE4-"
        self.s_score = "1VgiCFGJzA_99zSKUu_bylwxddtTIxMwe"
        self.g_score = "1iCWEofQjlHHrn-CeOtGJIkPKcYl1jLUh"

    def read(self, time_period="ten_days"):
        if time_period == "ten_days":
            self.ten_days()
        elif time_period == "one_month":
            self.one_month()
        else:
            print("We don't have data for that")
            return

        data = {"conn": pd_drive(self.connections),
                "data": pd_drive(self.data, parse_dates=["DATE"],
                                 infer_datetime_format=True),
                "embed": pd_drive(self.embeddings),
                "overall_score": pd_drive(self.daily_esg, parse_dates=["date"],
                                 infer_datetime_format=True, index_col="date"),
                "E_score": pd_drive(self.e_score, parse_dates=["date"],
                                 infer_datetime_format=True, index_col="date"),
                "S_score": pd_drive(self.s_score, parse_dates=["date"],
                                 infer_datetime_format=True, index_col="date"),
                "G_score": pd_drive(self.g_score, parse_dates=["date"],
                                 infer_datetime_format=True, index_col="date"),
                "ESG": pd_drive(self.avg_esg),
                }
        data["data"]["DATE"] = data["data"]["DATE"].dt.date
        return data
