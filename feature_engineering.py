"""
cleans up the raw online retail data and builds customer level features
(RFM style) so we can cluster customers later
"""

import pandas as pd
import numpy as np

RAW_PATH = "../data/Online_Retail.xlsx"
OUT_PATH = "../data/customer_features.csv"


def load_and_clean(path=RAW_PATH):
    df = pd.read_excel(path)
    print('raw shape:', df.shape)

    # get rid of rows with no customer id, cant use these for segmentation
    print('before:', len(df))
    df = df.dropna(subset=["CustomerID"])
    print('after dropping missing CustomerID:', len(df))

    # cancelled orders start with 'C' in invoice no, these are returns not real purchases
    print('before:', len(df))
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print('after removing cancellations:', len(df))

    # negative/zero qty or price = weird data entry stuff, drop it
    print('before:', len(df))
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print('after:', len(df))

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print('final clean shape:', df.shape)
    print('unique customers:', df['CustomerID'].nunique())
    return df


def build_customer_features(df):
    # reference date to calculate recency from - one day after last invoice in the data
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_df = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        TotalItems=("Quantity", "sum"),
    ).reset_index()

    # avg order value = total spent / number of orders
    customer_df["AvgOrderValue"] = customer_df["Monetary"] / customer_df["Frequency"]

    # tenure - how long theyve been a customer (days since first purchase)
    first_purchase = df.groupby("CustomerID")["InvoiceDate"].min()
    customer_df["TenureDays"] = customer_df["CustomerID"].map(
        lambda c: (snapshot_date - first_purchase[c]).days
    )

    print('customer feature table shape:', customer_df.shape)
    print(customer_df.describe())
    return customer_df


if __name__ == "__main__":
    df = load_and_clean()
    customer_df = build_customer_features(df)
    customer_df.to_csv(OUT_PATH, index=False)
    print('saved to', OUT_PATH)
